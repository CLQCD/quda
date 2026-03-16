#pragma once

#include <vector>
#include <color_spinor_field.h>

/**
 * @file eigen_checkpoint.h
 * @brief TRLM Eigensolver Checkpointing Utilities
 *
 * This module implements a robust, double-buffered, and incremental checkpointing 
 * system for the Thick Restarted Lanczos Method (TRLM).
 *
 * ============================================================================
 * Usage Guide
 * ============================================================================
 *
 * 1. Initialization:
 * Before the TRLM loop, initialize `int current_slot = -1;`.
 * Call `loadTRLMCheckpoint` to attempt restoring the solver state.
 * If successful, `current_slot` and solver state (kSpace, alpha, beta) will be updated.
 *
 * 2. Saving:
 * Call `saveTRLMCheckpoint` at two locations:
 * a) Inside the Lanczos step loop (frequency controlled by `save_interval`).
 * b) At the end of a restart cycle (to save the compressed Krylov space).
 *
 * 3. Incremental Logic:
 * The saver automatically detects if it is safe to perform an "Incremental Save"
 * (appending only new vectors to the existing file). 
 * Conditions for incremental save:
 * - `save_interval > 0`
 * - Target file belongs to the current restart cycle.
 * - Target file has no data holes (target_prev_k_step >= new_start).
 * If these conditions are not met, it falls back to a "Full Save" automatically.
 *
 * ============================================================================
 * Important Notes & Limitations
 * ============================================================================
 *
 * 1. MPI Configuration:
 * The checkpoint data is stored using a fixed-stride layout based on MPI ranks.
 * YOU CANNOT CHANGE THE NUMBER OF MPI PROCESSES between the save and load runs.
 * Attempting to resume with a different number of ranks will result in 
 * data corruption or load failure.
 *
 * 2. Save Interval Flexibility:
 * It is safe to change `chk_save_interval` between runs. The "Target Check" 
 * mechanism ensures that if the interval is reduced (creating potential holes), 
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

  struct TRLMCheckpointHeader {
    int version;
    int n_ranks;
    int restart_iter;
    int iter;
    int num_locked;
    int num_converged;
    int num_keep;
    int n_kr;
    int k_step;
  };

  void saveTRLMCheckpoint(const char* filename_base, 
                          std::vector<ColorSpinorField> &kSpace,
                          const std::vector<double> &alpha,
                          const std::vector<double> &beta,
                          int restart_iter, int iter, int num_locked, 
                          int num_converged, int num_keep, int n_kr,
                          int k_step,
                          int &current_slot,
                          int save_interval);

  bool loadTRLMCheckpoint(const char* filename_base, 
                          std::vector<ColorSpinorField> &kSpace,
                          std::vector<double> &alpha,
                          std::vector<double> &beta,
                          int &restart_iter, int &iter, int &num_locked, 
                          int &num_converged, int &num_keep, int n_kr,
                          int &k_step,
                          int &current_slot);

} // namespace quda