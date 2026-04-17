#pragma once

#include <vector>
#include <color_spinor_field.h>

/**
 * @file eigen_checkpoint.h
 * @brief Eigensolver checkpointing utilities
 *
 * This module implements double-buffered and incremental checkpointing for
 * eigensolvers.
 *
 * Current metadata schema version: 1
 * Required metadata tags:
 * - Layout = PARITY_SUBARRAY
 * - ChecksumScope = PER_VECTOR_GLOBAL
 * - ChecksumAlgo = CRC64-ECMA
 * - STATUS = COMPLETED
 *
 * ============================================================================
 * Usage Guide
 * ============================================================================
 *
 * 1. Initialization:
 * Before the solver loop, initialize `int current_slot = -1;`.
 * Call `loadCheckpoint` to attempt restoring the solver state.
 * If successful, `current_slot` and solver state (kSpace, alpha, beta) will be updated.
 *
 * 2. Saving:
 * Call `saveCheckpoint` at two locations:
 * a) Inside the solver iteration loop (frequency controlled by `save_interval`).
 * b) At the end of a restart/update cycle (to save the compressed working subspace).
 *
 * 3. Incremental Logic:
 * The saver automatically detects if it is safe to perform an "Incremental Save"
 * (append only new vectors to the existing target slot).
 * Conditions for incremental save:
 * - `save_interval > 0`
 * - `potential_start = k_step - 2 * save_interval` satisfies `potential_start >= num_keep`.
 * - Target slot belongs to the current restart cycle (`target_prev_restart == restart_iter`).
 * - Target slot has no holes (`target_prev_k_step >= potential_start`).
 * If these conditions are not met, it falls back to a "Full Save" automatically.
 *
 * ============================================================================
 * Notes
 * ============================================================================
 *
 * 1. MPI Configuration:
 * Checkpoints use a global subarray layout descriptor and can be restored with
 * a different process count, as long as the global checkerboard lattice shape
 * matches metadata (`GlobalX_CB`) and the site subset type matches (`IsFullSubset`).
 *
 * 2. Save Interval Flexibility:
 * It is safe to change `chk_save_interval` between runs. The target-slot checks
 * ensure that if the interval is reduced (creating potential holes),
 * the system automatically forces a full rewrite to ensure consistency.
 *
 * 3. File Structure:
 * - *.ckpt.latest      : Pointer file containing the ID (0 or 1) of the valid slot.
 * - *.ckpt.data.[0/1]  : Binary vector data (MPI-IO).
 * - *.ckpt.meta.[0/1]  : Text metadata (iteration counts, alpha/beta, etc).
 *
 * 4. Loop Boundary:
 * The save/load range is [0, k_step]. This includes the vector at index `k_step`,
 * which contains the residual vector required for the next Lanczos step.
 */

namespace quda {

   struct CheckpointHeader {
     int version;        // Metadata schema version.
     int n_ranks;        // Stored for diagnostics; not enforced on load.
     int restart_iter;   // Restart cycle index.
     int iter;           // Total solver iteration count.
     int num_locked;     // Number of locked eigenmodes.
     int num_converged;  // Number of converged eigenmodes.
     int num_keep;       // Number of kept vectors after restart.
     int n_kr;           // Krylov subspace size.
     int k_step;         // Last saved vector index (inclusive).
  };

   void saveCheckpoint(const char* filename_base,
                          std::vector<ColorSpinorField> &kSpace,
                          const std::vector<double> &alpha,
                          const std::vector<double> &beta,
                          int restart_iter, int iter, int num_locked, 
                          int num_converged, int num_keep, int n_kr,
                          int k_step,
                          int &current_slot,
                          int save_interval);

   bool loadCheckpoint(const char* filename_base,
                          std::vector<ColorSpinorField> &kSpace,
                          std::vector<double> &alpha,
                          std::vector<double> &beta,
                          int &restart_iter, int &iter, int &num_locked, 
                          int &num_converged, int &num_keep, int n_kr,
                          int &k_step,
                          int &current_slot);

} // namespace quda