#include <stdio.h>
#include <vector>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mpi.h>

#include <quda_internal.h>
#include <comm_quda.h>
#include <color_spinor_field.h>
#include <util_quda.h>

#include <eigen_checkpoint.h>

namespace quda
{

    // Checkpoint metadata schema version.
    static const int CKPT_VERSION = 1;
    static const char *CKPT_LAYOUT_PARITY_SUBARRAY = "PARITY_SUBARRAY";
    static const char *CKPT_CHECKSUM_SCOPE_PER_VECTOR_GLOBAL = "PER_VECTOR_GLOBAL";

    struct SubarrayLayout {
        bool valid = false;
        int global_x[4] = {0, 0, 0, 0};
        int local_x[4] = {0, 0, 0, 0};
        int starts[4] = {0, 0, 0, 0};
        size_t site_bytes = 0;
        MPI_Offset vec_bytes_global = 0;
    };

    static inline void crc64_table(uint64_t table[256])
    {
        static bool table_initialized = false;
        static uint64_t static_table[256];
        if (!table_initialized) {
            const uint64_t poly = 0x42F0E1EBA9EA3693ULL;
            for (int i = 0; i < 256; i++) {
                uint64_t crc = (uint64_t)i << 56;
                for (int j = 0; j < 8; j++) crc = (crc & 0x8000000000000000ULL) ? ((crc << 1) ^ poly) : (crc << 1);
                static_table[i] = crc;
            }
            table_initialized = true;
        }
        for (int i = 0; i < 256; i++) table[i] = static_table[i];
    }

    static inline uint64_t crc64_ecma_update(uint64_t crc, const void *data, size_t length)
    {
        uint64_t table[256];
        crc64_table(table);
        const unsigned char *p = static_cast<const unsigned char *>(data);
        for (size_t i = 0; i < length; i++) {
            uint8_t idx = (uint8_t)((crc >> 56) ^ p[i]);
            crc = table[idx] ^ (crc << 8);
        }
        return crc;
    }

    static bool computeVectorCRCFromFile(const char *data_filename, int vec_idx, MPI_Offset vec_bytes_total, uint64_t &crc_out)
    {
        MPI_File fh;
        int err = MPI_File_open(MPI_COMM_SELF, const_cast<char *>(data_filename), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        if (err != MPI_SUCCESS) return false;

        const size_t chunk = 4 * 1024 * 1024;
        std::vector<unsigned char> buffer(chunk);
        MPI_Offset offset = (MPI_Offset)vec_idx * vec_bytes_total;
        MPI_Offset left = vec_bytes_total;
        uint64_t crc = 0;
        while (left > 0) {
            int this_read = (int)((left > (MPI_Offset)chunk) ? chunk : left);
            MPI_Status status;
            err = MPI_File_read_at(fh, offset, buffer.data(), this_read, MPI_BYTE, &status);
            if (err != MPI_SUCCESS) {
                MPI_File_close(&fh);
                return false;
            }
            crc = crc64_ecma_update(crc, buffer.data(), (size_t)this_read);
            offset += this_read;
            left -= this_read;
        }

        MPI_File_close(&fh);
        crc_out = crc;
        return true;
    }

    // Build an MPI-IO subarray layout on a parity (checkerboard) lattice.
    static SubarrayLayout buildSubarrayLayoutParity(const ColorSpinorField &field)
    {
        SubarrayLayout l;
        if (field.SiteSubset() != QUDA_PARITY_SITE_SUBSET) return l;

        for (int d = 0; d < 4; d++) {
            int local_parity_dim = (d == 0) ? (field.full_dim(0) / 2) : field.full_dim(d);
            
            l.local_x[d]  = local_parity_dim;
            l.global_x[d] = local_parity_dim * comm_dim(d);
            l.starts[d]   = comm_coord(d) * local_parity_dim;
        }

        // Keep per-site byte size consistent with the runtime field representation.
        size_t local_volume = field.LocalVolume(); 
        if (local_volume == 0) return l;
        
        size_t vec_bytes = field.Bytes();
        l.site_bytes = vec_bytes / local_volume;
        if (l.site_bytes == 0) return l;

        MPI_Offset global_volume = (MPI_Offset)l.global_x[0] * l.global_x[1] * l.global_x[2] * l.global_x[3];
        l.vec_bytes_global = global_volume * (MPI_Offset)l.site_bytes;
        l.valid = true;
        return l;
    }

    // Collective write using a parity subarray file view.
    static int fileWriteVectorSubarrayParity(MPI_File fh, const void *buf, int vec_idx, MPI_Offset vec_bytes_total, const SubarrayLayout &layout, MPI_Offset internal_offset)
    {
        int gs[4] = {layout.global_x[3], layout.global_x[2], layout.global_x[1], (int)(layout.global_x[0] * (int)layout.site_bytes)};
        int ls[4] = {layout.local_x[3], layout.local_x[2], layout.local_x[1], (int)(layout.local_x[0] * (int)layout.site_bytes)};
        int st[4] = {layout.starts[3], layout.starts[2], layout.starts[1], (int)(layout.starts[0] * (int)layout.site_bytes)};

        MPI_Datatype filetype;
        int err = MPI_Type_create_subarray(4, gs, ls, st, MPI_ORDER_C, MPI_BYTE, &filetype);
        if (err != MPI_SUCCESS) return err;
        MPI_Type_commit(&filetype);

        MPI_Offset disp = (MPI_Offset)vec_idx * vec_bytes_total + internal_offset;
        err = MPI_File_set_view(fh, disp, MPI_BYTE, filetype, "native", MPI_INFO_NULL);
        if (err == MPI_SUCCESS) {
            MPI_Status status;
            err = MPI_File_write_all(fh, const_cast<void *>(buf), (int)(ls[0] * ls[1] * ls[2] * ls[3]), MPI_BYTE, &status);
        }
        MPI_Type_free(&filetype);
        return err;
    }

    // Collective read using a parity subarray file view.
    static int fileReadVectorSubarrayParity(MPI_File fh, void *buf, int vec_idx, MPI_Offset vec_bytes_total, const SubarrayLayout &layout, MPI_Offset internal_offset)
    {
        int gs[4] = {layout.global_x[3], layout.global_x[2], layout.global_x[1], (int)(layout.global_x[0] * (int)layout.site_bytes)};
        int ls[4] = {layout.local_x[3], layout.local_x[2], layout.local_x[1], (int)(layout.local_x[0] * (int)layout.site_bytes)};
        int st[4] = {layout.starts[3], layout.starts[2], layout.starts[1], (int)(layout.starts[0] * (int)layout.site_bytes)};

        MPI_Datatype filetype;
        int err = MPI_Type_create_subarray(4, gs, ls, st, MPI_ORDER_C, MPI_BYTE, &filetype);
        if (err != MPI_SUCCESS) return err;
        MPI_Type_commit(&filetype);

        MPI_Offset disp = (MPI_Offset)vec_idx * vec_bytes_total + internal_offset;
        err = MPI_File_set_view(fh, disp, MPI_BYTE, filetype, "native", MPI_INFO_NULL);
        if (err == MPI_SUCCESS) {
            MPI_Status status;
            err = MPI_File_read_all(fh, buf, (int)(ls[0] * ls[1] * ls[2] * ls[3]), MPI_BYTE, &status);
        }
        MPI_Type_free(&filetype);
        return err;
    }

    void saveCheckpoint(const char* filename_base,
                          std::vector<ColorSpinorField> &kSpace,
                          const std::vector<double> &alpha,
                          const std::vector<double> &beta,
                          int restart_iter, int iter, int num_locked, 
                          int num_converged, int num_keep, int n_kr,
                          int k_step,
                          int &current_slot,
                          int save_interval)
    {
        int rank = comm_rank();
        int size = comm_size();
    
        int target_slot = (current_slot + 1) % 2;

        char data_filename[256];
        char meta_filename[256];
        char meta_tmp_filename[256];
        char latest_filename[256];
        char latest_tmp_filename[256]; 

        sprintf(data_filename, "%s.ckpt.data.%d", filename_base, target_slot);
        sprintf(meta_filename, "%s.ckpt.meta.%d", filename_base, target_slot);
        sprintf(meta_tmp_filename, "%s.ckpt.meta.%d.tmp", filename_base, target_slot);
        sprintf(latest_filename, "%s.ckpt.latest", filename_base);
        sprintf(latest_tmp_filename, "%s.ckpt.latest.tmp", filename_base);

        int target_prev_k_step = -1;
        int target_prev_restart = -1;
        int target_prev_checksum_count = 0;
        std::vector<uint64_t> prev_vector_checksums;

        if (rank == 0) {
            FILE* fp = fopen(meta_filename, "r");
            if (fp) {
                char line[256];
                bool in_checksum_section = false;
                int checksum_idx = 0;
                while (fgets(line, sizeof(line), fp)) {
                    if (sscanf(line, "RestartIter: %d", &target_prev_restart) == 1) continue;
                    if (sscanf(line, "KStep: %d", &target_prev_k_step) == 1) continue;
                    if (sscanf(line, "ChecksumCount: %d", &target_prev_checksum_count) == 1) {
                        prev_vector_checksums.resize((size_t)target_prev_checksum_count);
                        continue;
                    }
                    if (strncmp(line, "Checksum:", 9) == 0) {
                        in_checksum_section = true;
                        continue;
                    }
                    if (in_checksum_section && checksum_idx < (int)prev_vector_checksums.size()) {
                        unsigned long long tmp = 0;
                        if (sscanf(line, "%llx", &tmp) == 1) {
                            prev_vector_checksums[checksum_idx++] = (uint64_t)tmp;
                        }
                    }
                    if (strncmp(line, "STATUS:", 7) == 0) break;
                }
                fclose(fp);
            }
        }
    
        int target_info[2] = {target_prev_restart, target_prev_k_step};
        MPI_Bcast(target_info, 2, MPI_INT, 0, MPI_COMM_WORLD);
        target_prev_restart = target_info[0];
        target_prev_k_step = target_info[1];

        int write_start = 0;
        
        if (save_interval > 0) {
            int gap = 2 * save_interval;
            int potential_start = k_step - gap;
            
            bool safe_to_append = true;
            
            if (potential_start < num_keep) safe_to_append = false;
            if (target_prev_restart != restart_iter) safe_to_append = false;
            if (target_prev_k_step < potential_start) safe_to_append = false;

            if (safe_to_append) write_start = potential_start;
            else write_start = 0;
        }

        // Use a CPU mirror field for MPI-IO.
        ColorSpinorParam param(kSpace[0]);
        param.location = QUDA_CPU_FIELD_LOCATION;
        param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        param.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField temp_cpu(param); 

        bool is_full = (temp_cpu.SiteSubset() == QUDA_FULL_SITE_SUBSET);
        // For full fields, the file layout is defined on the even checkerboard.
        SubarrayLayout subarray_layout = buildSubarrayLayoutParity(is_full ? temp_cpu.Even() : temp_cpu);
        
        if (!subarray_layout.valid) {
            if (rank == 0) warningQuda("Checkpoint: failed to build parity layout");
            return;
        }

        MPI_Offset vec_bytes_total = subarray_layout.vec_bytes_global * (is_full ? 2 : 1);
        
        MPI_File fh;
        int err = MPI_File_open(MPI_COMM_WORLD, data_filename, 
                                MPI_MODE_CREATE | MPI_MODE_WRONLY, 
                                MPI_INFO_NULL, &fh);
        
        if (err != MPI_SUCCESS) {
            if (rank == 0) warningQuda("Checkpoint: failed to open data file with MPI-IO");
            return;
        }

        if (write_start == 0) {
            MPI_Offset total_size = (MPI_Offset)(k_step + 1) * vec_bytes_total;
            MPI_File_set_size(fh, total_size);
        }

        bool io_ok = true;
        for(int i = write_start; i <= k_step; i++) {
            temp_cpu = kSpace[i];
            
            if (is_full) {
                err = fileWriteVectorSubarrayParity(fh, temp_cpu.Even().data(), i, vec_bytes_total, subarray_layout, 0);
                if (err != MPI_SUCCESS) { io_ok = false; break; }
                err = fileWriteVectorSubarrayParity(fh, temp_cpu.Odd().data(), i, vec_bytes_total, subarray_layout, subarray_layout.vec_bytes_global);
                if (err != MPI_SUCCESS) { io_ok = false; break; }
            } else {
                err = fileWriteVectorSubarrayParity(fh, temp_cpu.data(), i, vec_bytes_total, subarray_layout, 0);
                if (err != MPI_SUCCESS) { io_ok = false; break; }
            }
        }

        if (!io_ok) {
            if (rank == 0) warningQuda("Checkpoint: MPI-IO subarray write failed");
            MPI_File_close(&fh);
            return;
        }

        MPI_File_close(&fh);
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<uint64_t> vector_checksums(k_step + 1, 0);
        bool checksum_compute_ok = true;
        if (rank == 0) {
            for (int i = write_start; i <= k_step; i++) {
                if (!computeVectorCRCFromFile(data_filename, i, vec_bytes_total, vector_checksums[i])) {
                    checksum_compute_ok = false;
                    break;
                }
            }
        }
        int checksum_ok_int = checksum_compute_ok ? 1 : 0;
        MPI_Bcast(&checksum_ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!checksum_ok_int) {
            if (rank == 0) warningQuda("Checkpoint: failed to compute CRC64 checksums from data file");
            return;
        }

        bool meta_commit_ok = false;
        bool latest_commit_ok = false;
        if (rank == 0) {
            FILE* fp = fopen(meta_tmp_filename, "w");
            if (fp) {
                fprintf(fp, "Version: %d\n", CKPT_VERSION);
                fprintf(fp, "NRanks: %d\n", size);
                fprintf(fp, "RestartIter: %d\n", restart_iter);
                fprintf(fp, "TotalIter: %d\n", iter);
                fprintf(fp, "NumLocked: %d\n", num_locked);
                fprintf(fp, "NumConverged: %d\n", num_converged);
                fprintf(fp, "NumKeep: %d\n", num_keep);
                fprintf(fp, "NKr: %d\n", n_kr);
                fprintf(fp, "KStep: %d\n", k_step);
                fprintf(fp, "Layout: %s\n", CKPT_LAYOUT_PARITY_SUBARRAY);
                fprintf(fp, "ChecksumScope: %s\n", CKPT_CHECKSUM_SCOPE_PER_VECTOR_GLOBAL);
                fprintf(fp, "GlobalX_CB: %d %d %d %d\n", subarray_layout.global_x[0], subarray_layout.global_x[1],
                        subarray_layout.global_x[2], subarray_layout.global_x[3]);
                fprintf(fp, "SiteBytes: %zu\n", subarray_layout.site_bytes);
                fprintf(fp, "IsFullSubset: %d\n", is_full ? 1 : 0);
                
                fprintf(fp, "Alpha:\n");
                for(int i=0; i<n_kr; i++) fprintf(fp, "%.16e\n", alpha[i]);
                
                fprintf(fp, "Beta:\n");
                for(int i=0; i<n_kr; i++) fprintf(fp, "%.16e\n", beta[i]);

                fprintf(fp, "ChecksumAlgo: CRC64-ECMA\n");
                fprintf(fp, "ChecksumCount: %d\n", k_step + 1);
                fprintf(fp, "Checksum:\n");
                for (int i = 0; i <= k_step; i++) {
                    uint64_t crc = 0;
                    // Reuse old checksums for untouched vectors in incremental mode.
                    if (i < write_start && write_start > 0) {
                        if (i < (int)prev_vector_checksums.size()) crc = prev_vector_checksums[i];
                        else warningQuda("Checkpoint: missing previous checksum for vector %d", i);
                    } else {
                        crc = vector_checksums[i];
                    }
                    fprintf(fp, "%016llx\n", (unsigned long long)crc);
                }
                
                fprintf(fp, "STATUS: COMPLETED\n");

                bool meta_io_ok = (fflush(fp) == 0);
                if (meta_io_ok) meta_io_ok = (fsync(fileno(fp)) == 0);
                if (fclose(fp) != 0) meta_io_ok = false;

                if (!meta_io_ok) {
                    warningQuda("Checkpoint: failed to flush metadata file");
                } else if (rename(meta_tmp_filename, meta_filename) != 0) {
                    warningQuda("Checkpoint: failed to publish metadata file");
                } else {
                    meta_commit_ok = true;
                }
            } else {
                warningQuda("Checkpoint: failed to write metadata file");
            }
            if (meta_commit_ok) {
                FILE* fp_ptr = fopen(latest_tmp_filename, "w");
                if (fp_ptr) {
                    bool latest_io_ok = (fprintf(fp_ptr, "%d\n", target_slot) >= 0);
                    if (latest_io_ok) latest_io_ok = (fflush(fp_ptr) == 0);
                    if (latest_io_ok) latest_io_ok = (fsync(fileno(fp_ptr)) == 0);
                    if (fclose(fp_ptr) != 0) latest_io_ok = false;

                    if (!latest_io_ok) {
                        warningQuda("Checkpoint: failed to flush latest pointer file");
                    } else if (rename(latest_tmp_filename, latest_filename) != 0) {
                        warningQuda("Checkpoint: failed to publish latest pointer file");
                    } else {
                        latest_commit_ok = true;
                    }
                } else {
                    warningQuda("Checkpoint: failed to write latest pointer file");
                }
            }

            if (meta_commit_ok && latest_commit_ok) {
                logQuda(QUDA_VERBOSE, "Checkpoint saved to slot %d. Range: [%d, %d]. Mode: %s\n",
                        target_slot, write_start, k_step, (write_start > 0) ? "Incremental" : "Full");
            }
        }

        int commit_ok_int = (meta_commit_ok && latest_commit_ok) ? 1 : 0;
        MPI_Bcast(&commit_ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!commit_ok_int) return;

        current_slot = target_slot;
    }

    bool loadCheckpoint(const char* filename_base,
                          std::vector<ColorSpinorField> &kSpace,
                          std::vector<double> &alpha,
                          std::vector<double> &beta,
                          int &restart_iter, int &iter, int &num_locked, 
                          int &num_converged, int &num_keep, int n_kr,
                          int &k_step,
                          int &current_slot)
    {
        int rank = comm_rank();
        char data_filename[256];
        char meta_filename[256];
        char latest_filename[256];

        sprintf(latest_filename, "%s.ckpt.latest", filename_base);

        int slot_tmp = 0;
        bool has_ckpt = false;

        if (rank == 0) {
            FILE* fp = fopen(latest_filename, "r");
            if (fp) {
                if (fscanf(fp, "%d", &slot_tmp) == 1) has_ckpt = true;
                fclose(fp);
            }
        }
        
        MPI_Bcast(&has_ckpt, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        if (!has_ckpt) return false;

        MPI_Bcast(&slot_tmp, 1, MPI_INT, 0, MPI_COMM_WORLD);
        current_slot = slot_tmp;

        sprintf(data_filename, "%s.ckpt.data.%d", filename_base, current_slot);
        sprintf(meta_filename, "%s.ckpt.meta.%d", filename_base, current_slot);

        bool file_exists = false;
        if (rank == 0) {
            struct stat buffer;
            if (stat(meta_filename, &buffer) == 0) file_exists = true;
        }
        MPI_Bcast(&file_exists, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        if (!file_exists) return false;

        CheckpointHeader header;
        memset(&header, 0, sizeof(header));
        std::vector<double> temp_alpha(n_kr), temp_beta(n_kr);
        std::vector<uint64_t> file_checksums;
        int checksum_count = 0;
        bool checksum_ok = false;
        bool meta_is_complete = false;
        bool is_parity_layout = false;
        bool is_per_vector_global_checksum = false;
        int meta_global_x_cb[4] = {0, 0, 0, 0};
        size_t meta_site_bytes = 0;
        int meta_is_full = 0;

        if (rank == 0) {
            FILE* fp = fopen(meta_filename, "r");
            if (fp) {
                fscanf(fp, "Version: %d\n", &header.version);
                fscanf(fp, "NRanks: %d\n", &header.n_ranks);
                fscanf(fp, "RestartIter: %d\n", &header.restart_iter);
                fscanf(fp, "TotalIter: %d\n", &header.iter);
                fscanf(fp, "NumLocked: %d\n", &header.num_locked);
                fscanf(fp, "NumConverged: %d\n", &header.num_converged);
                fscanf(fp, "NumKeep: %d\n", &header.num_keep);
                fscanf(fp, "NKr: %d\n", &header.n_kr);
                fscanf(fp, "KStep: %d\n", &header.k_step);

                if (header.version != CKPT_VERSION) {
                    warningQuda("Checkpoint: unsupported version %d (expected %d)", header.version, CKPT_VERSION);
                    fclose(fp); meta_is_complete = false; goto meta_parse_done;
                }

                char line[256];
                bool have_alpha_header = false;
                while (fgets(line, sizeof(line), fp)) {
                    char layout_buf[64]; char checksum_scope_buf[64];
                    if (sscanf(line, "Layout: %63s", layout_buf) == 1) {
                        if (strcmp(layout_buf, CKPT_LAYOUT_PARITY_SUBARRAY) == 0) is_parity_layout = true; continue;
                    }
                    if (sscanf(line, "ChecksumScope: %63s", checksum_scope_buf) == 1) {
                        if (strcmp(checksum_scope_buf, CKPT_CHECKSUM_SCOPE_PER_VECTOR_GLOBAL) == 0) is_per_vector_global_checksum = true; continue;
                    }
                    if (sscanf(line, "GlobalX_CB: %d %d %d %d", &meta_global_x_cb[0], &meta_global_x_cb[1], &meta_global_x_cb[2], &meta_global_x_cb[3]) == 4) continue;
                    if (sscanf(line, "SiteBytes: %zu", &meta_site_bytes) == 1) continue;
                    if (sscanf(line, "IsFullSubset: %d", &meta_is_full) == 1) continue;
                    if (strncmp(line, "Alpha:", 6) == 0) { have_alpha_header = true; break; }
                }

                if (!have_alpha_header || !is_parity_layout || !is_per_vector_global_checksum) {
                    warningQuda("Checkpoint: malformed metadata in %s", meta_filename);
                    fclose(fp); meta_is_complete = false; goto meta_parse_done;
                }

                std::vector<double> file_alpha(header.n_kr, 0.0);
                std::vector<double> file_beta(header.n_kr, 0.0);
                
                for(int i=0; i<header.n_kr; i++) fscanf(fp, "%lf\n", &file_alpha[i]);
                fscanf(fp, "Beta:\n");
                for(int i=0; i<header.n_kr; i++) fscanf(fp, "%lf\n", &file_beta[i]);

                char algo_buf[64];
                if (fscanf(fp, "ChecksumAlgo: %63s\n", algo_buf) == 1 && fscanf(fp, "ChecksumCount: %d\n", &checksum_count) == 1) {
                    if (strcmp(algo_buf, "CRC64-ECMA") == 0 && checksum_count == (header.k_step + 1)) {
                        fscanf(fp, "Checksum:\n"); checksum_ok = true;
                    }
                }

                if (checksum_ok) {
                    file_checksums.resize((size_t)checksum_count);
                    for (size_t i = 0; i < file_checksums.size(); i++) {
                        unsigned long long tmp = 0;
                        if (fscanf(fp, "%llx\n", &tmp) != 1) { checksum_ok = false; break; }
                        file_checksums[i] = (uint64_t)tmp;
                    }
                }
                
                char status_buffer[32];
                if (checksum_ok && fscanf(fp, "STATUS: %s\n", status_buffer) == 1 && strcmp(status_buffer, "COMPLETED") == 0) meta_is_complete = true;
                
                if (meta_is_complete) {
                    int copy_len = std::min(header.n_kr, n_kr);
                    for(int i = 0; i < copy_len; i++) {
                        temp_alpha[i] = file_alpha[i];
                        temp_beta[i]  = file_beta[i];
                    }
                }

                fclose(fp);
            }
        }

meta_parse_done:

        MPI_Bcast(&meta_is_complete, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        if (!meta_is_complete) return false;

        MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(meta_global_x_cb, 4, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&meta_site_bytes, sizeof(meta_site_bytes), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&meta_is_full, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) { alpha = temp_alpha; beta = temp_beta; }
        MPI_Bcast(alpha.data(), n_kr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(beta.data(), n_kr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (n_kr <= header.k_step) {
            if (rank == 0) warningQuda("Checkpoint: Runtime n_kr (%d) is too small to hold restored k_step (%d)", n_kr, header.k_step);
            return false;
        }

        restart_iter = header.restart_iter; iter = header.iter;
        num_locked = header.num_locked; num_converged = header.num_converged;
        num_keep = header.num_keep; k_step = header.k_step;

        MPI_File fh;
        int err = MPI_File_open(MPI_COMM_WORLD, data_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        if (err != MPI_SUCCESS) return false;

        ColorSpinorParam param(kSpace[0]);
        param.location = QUDA_CPU_FIELD_LOCATION;
        param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        param.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField temp_cpu(param);

        bool is_full = (temp_cpu.SiteSubset() == QUDA_FULL_SITE_SUBSET);
        if ((is_full ? 1 : 0) != meta_is_full) {
            if (rank == 0) warningQuda("Checkpoint: site subset mismatch between run and metadata");
            MPI_File_close(&fh); return false;
        }

        SubarrayLayout subarray_layout = buildSubarrayLayoutParity(is_full ? temp_cpu.Even() : temp_cpu);
        if (!subarray_layout.valid) {
            if (rank == 0) warningQuda("Checkpoint: failed to build parity layout for restore");
            MPI_File_close(&fh); return false;
        }

        if (meta_site_bytes != 0 && meta_site_bytes != subarray_layout.site_bytes) {
            if (rank == 0) warningQuda("Checkpoint: site-bytes mismatch between metadata and runtime field");
            MPI_File_close(&fh); return false;
        }

        for (int d = 0; d < 4; d++) {
            if (meta_global_x_cb[d] != 0 && meta_global_x_cb[d] != subarray_layout.global_x[d]) {
                if (rank == 0) warningQuda("Checkpoint: parity lattice mismatch");
                MPI_File_close(&fh); return false;
            }
        }
        
        MPI_Offset vec_bytes_total = subarray_layout.vec_bytes_global * (is_full ? 2 : 1);

        bool local_crc_ok = true;
        for(int i = 0; i <= k_step; i++) {
            if (is_full) {
                err = fileReadVectorSubarrayParity(fh, temp_cpu.Even().data(), i, vec_bytes_total, subarray_layout, 0);
                if (err != MPI_SUCCESS) { local_crc_ok = false; break; }
                err = fileReadVectorSubarrayParity(fh, temp_cpu.Odd().data(), i, vec_bytes_total, subarray_layout, subarray_layout.vec_bytes_global);
                if (err != MPI_SUCCESS) { local_crc_ok = false; break; }
            } else {
                err = fileReadVectorSubarrayParity(fh, temp_cpu.data(), i, vec_bytes_total, subarray_layout, 0);
                if (err != MPI_SUCCESS) { local_crc_ok = false; break; }
            }
            // Copy restored vector back to the runtime field (GPU-backed in normal runs).
            kSpace[i] = temp_cpu;
        }

        MPI_File_close(&fh);

        bool global_crc_ok_from_file = true;
        if (rank == 0) {
            for (int i = 0; i <= k_step; i++) {
                uint64_t crc = 0;
                if (!computeVectorCRCFromFile(data_filename, i, vec_bytes_total, crc) || crc != file_checksums[i]) {
                    global_crc_ok_from_file = false; break;
                }
            }
        }
        
        int global_crc_ok_file_int = global_crc_ok_from_file ? 1 : 0;
        MPI_Bcast(&global_crc_ok_file_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!global_crc_ok_file_int) local_crc_ok = false;

        int global_crc_ok = 0; int local_ok_int = local_crc_ok ? 1 : 0;
        MPI_Allreduce(&local_ok_int, &global_crc_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (!global_crc_ok) return false;

        if (rank == 0) logQuda(QUDA_SUMMARIZE, "Checkpoint loaded successfully from slot %d\n", current_slot);
        
        return true;
    }

} // namespace quda