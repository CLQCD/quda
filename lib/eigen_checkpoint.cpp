#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <mpi.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <util_quda.h>

#include <eigen_checkpoint.h>

namespace quda
{

    static uint64_t crc64_ecma(const void *data, size_t length)
    {
        static bool table_initialized = false;
        static uint64_t table[256];
        const uint64_t poly = 0x42F0E1EBA9EA3693ULL;

        if (!table_initialized) {
            for (int i = 0; i < 256; i++) {
                uint64_t crc = (uint64_t)i << 56;
                for (int j = 0; j < 8; j++) {
                    if (crc & 0x8000000000000000ULL) {
                        crc = (crc << 1) ^ poly;
                    } else {
                        crc <<= 1;
                    }
                }
                table[i] = crc;
            }
            table_initialized = true;
        }

        uint64_t crc = 0;
        const unsigned char *p = static_cast<const unsigned char *>(data);
        for (size_t i = 0; i < length; i++) {
            uint8_t idx = (uint8_t)((crc >> 56) ^ p[i]);
            crc = table[idx] ^ (crc << 8);
        }
        return crc;
    }

    void saveTRLMCheckpoint(const char* filename_base, 
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
        int target_prev_checksum_nranks = 0;
        int target_prev_checksum_count = 0;
        std::vector<uint64_t> prev_all_checksums;

        if (rank == 0) {
            FILE* fp = fopen(meta_filename, "r");
            if (fp) {
                char line[256];
                bool in_checksum_section = false;
                int checksum_idx = 0;
                while (fgets(line, sizeof(line), fp)) {
                    if (sscanf(line, "RestartIter: %d", &target_prev_restart) == 1) continue;
                    if (sscanf(line, "KStep: %d", &target_prev_k_step) == 1) continue;
                    if (sscanf(line, "ChecksumNRanks: %d", &target_prev_checksum_nranks) == 1) continue;
                    if (sscanf(line, "ChecksumCount: %d", &target_prev_checksum_count) == 1) {
                        prev_all_checksums.resize((size_t)target_prev_checksum_nranks * target_prev_checksum_count);
                        continue;
                    }
                    if (strncmp(line, "Checksum:", 9) == 0) {
                        in_checksum_section = true;
                        continue;
                    }
                    if (in_checksum_section && checksum_idx < (int)prev_all_checksums.size()) {
                        unsigned long long tmp = 0;
                        if (sscanf(line, "%llx", &tmp) == 1) {
                            prev_all_checksums[checksum_idx++] = (uint64_t)tmp;
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

            if (safe_to_append) {
                write_start = potential_start;
            } else {
                write_start = 0;
            }
        }

        ColorSpinorParam param(kSpace[0]);
        param.location = QUDA_CPU_FIELD_LOCATION;
        param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        param.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField temp_cpu(param);
        
        MPI_Offset vec_bytes = temp_cpu.Bytes();
        MPI_Offset local_block_size = n_kr * vec_bytes; 
        MPI_Offset my_offset = (MPI_Offset)rank * local_block_size;

        MPI_File fh;
        int err = MPI_File_open(MPI_COMM_WORLD, data_filename, 
                                MPI_MODE_CREATE | MPI_MODE_WRONLY, 
                                MPI_INFO_NULL, &fh);
        
        if (err != MPI_SUCCESS) {
            if (rank == 0) warningQuda("MPI_File_open failed for checkpoint data file");
            return;
        }

        std::vector<uint64_t> local_checksums(k_step + 1, 0);

        MPI_Status io_status;
        bool io_ok = true;
        for(int i = write_start; i <= k_step; i++) {
            temp_cpu = kSpace[i];
            MPI_Offset current_vec_offset = my_offset + i * vec_bytes;
            err = MPI_File_write_at(fh, current_vec_offset, temp_cpu.data(), vec_bytes, MPI_BYTE, &io_status);
            if (err != MPI_SUCCESS) { io_ok = false; break; }
            local_checksums[i] = crc64_ecma(temp_cpu.data(), (size_t)vec_bytes);
        }

        if (!io_ok) {
            if (rank == 0) warningQuda("MPI_File_write_at failed for checkpoint data file");
            MPI_File_close(&fh);
            return;
        }

        MPI_File_close(&fh);

        MPI_Barrier(MPI_COMM_WORLD); 

        std::vector<uint64_t> all_checksums;
        if (rank == 0) all_checksums.resize((size_t)size * (k_step + 1));
        MPI_Gather(local_checksums.data(), k_step + 1, MPI_UINT64_T,
                   (rank == 0) ? all_checksums.data() : nullptr, k_step + 1, MPI_UINT64_T,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            FILE* fp = fopen(meta_tmp_filename, "w");
            if (fp) {
                fprintf(fp, "Version: 1\n");
                fprintf(fp, "NRanks: %d\n", size);
                fprintf(fp, "RestartIter: %d\n", restart_iter);
                fprintf(fp, "TotalIter: %d\n", iter);
                fprintf(fp, "NumLocked: %d\n", num_locked);
                fprintf(fp, "NumConverged: %d\n", num_converged);
                fprintf(fp, "NumKeep: %d\n", num_keep);
                fprintf(fp, "NKr: %d\n", n_kr);
                fprintf(fp, "KStep: %d\n", k_step);
                
                fprintf(fp, "Alpha:\n");
                for(int i=0; i<n_kr; i++) fprintf(fp, "%.16e\n", alpha[i]);
                
                fprintf(fp, "Beta:\n");
                for(int i=0; i<n_kr; i++) fprintf(fp, "%.16e\n", beta[i]);

                fprintf(fp, "ChecksumAlgo: CRC64-ECMA\n");
                fprintf(fp, "ChecksumNRanks: %d\n", size);
                fprintf(fp, "ChecksumCount: %d\n", k_step + 1);
                fprintf(fp, "Checksum:\n");
                for (int r = 0; r < size; r++) {
                    for (int i = 0; i <= k_step; i++) {
                        uint64_t crc;
                        if (i < write_start && write_start > 0) {
                            crc = prev_all_checksums[(size_t)r * target_prev_checksum_count + i];
                        } else {
                            crc = all_checksums[(size_t)r * (k_step + 1) + i];
                        }
                        fprintf(fp, "%016llx\n", (unsigned long long)crc);
                    }
                }
                
                fprintf(fp, "STATUS: COMPLETED\n");
                fflush(fp);
                fsync(fileno(fp));
                fclose(fp);
            } else {
                warningQuda("Failed to write checkpoint metadata file");
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            rename(meta_tmp_filename, meta_filename);
            
            FILE* fp_ptr = fopen(latest_tmp_filename, "w");
            if (fp_ptr) {
                fprintf(fp_ptr, "%d\n", target_slot);
                fflush(fp_ptr);
                fsync(fileno(fp_ptr));
                fclose(fp_ptr);
                
                rename(latest_tmp_filename, latest_filename);
                
                logQuda(QUDA_VERBOSE, "Checkpoint saved to Slot %d. Range: [%d, %d]. Mode: %s\n", 
                        target_slot, write_start, k_step, 
                        (write_start > 0) ? "Incremental" : "Full");
            } else {
                warningQuda("Failed to write checkpoint pointer tmp file");
            }
        }

        current_slot = target_slot;
    }

    bool loadTRLMCheckpoint(const char* filename_base, 
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
                if (fscanf(fp, "%d", &slot_tmp) == 1) {
                    has_ckpt = true;
                }
                fclose(fp);
            } else {
                logQuda(QUDA_VERBOSE, "Checkpoint pointer file %s not found. Starting fresh run.\n", latest_filename);
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

        TRLMCheckpointHeader header;
        std::vector<double> temp_alpha(n_kr), temp_beta(n_kr);
        std::vector<uint64_t> file_checksums;
        int checksum_nranks = 0;
        int checksum_count = 0;
        bool checksum_ok = false;
        bool meta_is_complete = false;

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

                fscanf(fp, "Alpha:\n");
                int read_n_kr = header.n_kr; 
                if(read_n_kr == n_kr) {
                    for(int i=0; i<n_kr; i++) fscanf(fp, "%lf\n", &temp_alpha[i]);
                    fscanf(fp, "Beta:\n");
                    for(int i=0; i<n_kr; i++) fscanf(fp, "%lf\n", &temp_beta[i]);

                    char algo_buf[64];
                    if (fscanf(fp, "ChecksumAlgo: %63s\n", algo_buf) == 1 &&
                        fscanf(fp, "ChecksumNRanks: %d\n", &checksum_nranks) == 1 &&
                        fscanf(fp, "ChecksumCount: %d\n", &checksum_count) == 1) {
                        if (strcmp(algo_buf, "CRC64-ECMA") == 0 && checksum_nranks == comm_size() && checksum_count == (header.k_step + 1)) {
                            fscanf(fp, "Checksum:\n");
                            checksum_ok = true;
                        } else {
                            warningQuda("Checksum metadata mismatch in %s", meta_filename);
                        }
                    }

                    if (checksum_ok) {
                        file_checksums.resize((size_t)checksum_nranks * checksum_count);
                        for (size_t i = 0; i < file_checksums.size(); i++) {
                            unsigned long long tmp = 0;
                            if (fscanf(fp, "%llx\n", &tmp) != 1) { checksum_ok = false; break; }
                            file_checksums[i] = (uint64_t)tmp;
                        }
                    }
                    
                    char status_buffer[32];
                    if (checksum_ok && fscanf(fp, "STATUS: %s\n", status_buffer) == 1) {
                        if (strcmp(status_buffer, "COMPLETED") == 0) {
                            meta_is_complete = true;
                        }
                    }
                    
                    if (!meta_is_complete) {
                        warningQuda("Metadata file %s is incomplete. Checkpoint corrupted.", meta_filename);
                    }
                } else {
                    warningQuda("Checkpoint n_kr mismatch (%d vs %d).", read_n_kr, n_kr);
                }
                fclose(fp);
            } else {
                warningQuda("Could not open metadata file %s", meta_filename);
            }
        }

        MPI_Bcast(&meta_is_complete, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        if (!meta_is_complete) return false;

        MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            alpha = temp_alpha;
            beta = temp_beta;
        }
        MPI_Bcast(alpha.data(), n_kr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(beta.data(), n_kr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (header.n_kr != n_kr) return false;
        if (header.n_ranks != comm_size()) return false;

        restart_iter = header.restart_iter;
        iter = header.iter;
        num_locked = header.num_locked;
        num_converged = header.num_converged;
        num_keep = header.num_keep;
        k_step = header.k_step;

        MPI_File fh;
        int err = MPI_File_open(MPI_COMM_WORLD, data_filename, 
                                MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        
        if (err != MPI_SUCCESS) {
            if(rank == 0) warningQuda("Could not open data file %s", data_filename);
            return false;
        }

        ColorSpinorParam param(kSpace[0]);
        param.location = QUDA_CPU_FIELD_LOCATION;
        param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        param.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField temp_cpu(param);
        
        MPI_Offset vec_bytes = temp_cpu.Bytes();
        MPI_Offset local_block_size = n_kr * vec_bytes;
        MPI_Offset my_offset = (MPI_Offset)rank * local_block_size;

        std::vector<uint64_t> expected_local(k_step + 1, 0);
        MPI_Scatter((rank == 0) ? file_checksums.data() : nullptr, k_step + 1, MPI_UINT64_T,
                expected_local.data(), k_step + 1, MPI_UINT64_T,
                0, MPI_COMM_WORLD);

        MPI_Status io_status;
        bool local_crc_ok = true;
        for(int i = 0; i <= k_step; i++) {
            MPI_Offset current_vec_offset = my_offset + i * vec_bytes;
            err = MPI_File_read_at(fh, current_vec_offset, temp_cpu.data(), vec_bytes, MPI_BYTE, &io_status);
            if (err != MPI_SUCCESS) { local_crc_ok = false; break; }
            uint64_t crc = crc64_ecma(temp_cpu.data(), (size_t)vec_bytes);
            if (expected_local[i] != 0 && expected_local[i] != crc) local_crc_ok = false;
            kSpace[i] = temp_cpu;
        }

        MPI_File_close(&fh);

        int global_crc_ok = 0;
        int local_ok_int = local_crc_ok ? 1 : 0;
        MPI_Allreduce(&local_ok_int, &global_crc_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        if (!global_crc_ok) {
            if (rank == 0) warningQuda("Checkpoint CRC64 verification failed.");
            return false;
        }

        if (rank == 0) {
            logQuda(QUDA_SUMMARIZE, "Checkpoint loaded successfully from Slot %d (Restart: %d, Step: %d)\n", current_slot, restart_iter, k_step);
        }
        
        return true;
    }

} // namespace quda