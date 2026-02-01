#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <memory>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <eigensolve_quda.h>
#include <eigen_helper.h>

#include <reliable_updates.h>
#include <invert_x_update.h>

namespace quda
{

  BlockCG::BlockCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, const DiracMatrix &matEig,
         SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param)
  {
  }

  BlockCG::~BlockCG() { destroyDeflationSpace(); }

  void BlockCG::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);

    if (!init || r.size() != b.size()) {
      getProfile().TPSTART(QUDA_PROFILE_INIT);

      resize(r, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);
      resize(y, b.size(), QUDA_NULL_FIELD_CREATE, b[0]);

      // sloppy fields
      ColorSpinorParam csParam(x[0]);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);
      resize(p, b.size(), csParam);
      resize(Ap, b.size(), csParam);
      resize(rnew, b.size(), csParam);

      if (param.precision != param.precision_sloppy) {
        resize(r_sloppy, b.size(), csParam);
      } else {
        create_alias(r_sloppy, r);
      }
      param.use_sloppy_partial_accumulator = false; // hard-code precise accumulation
      if (param.use_sloppy_partial_accumulator) resize(x_sloppy, b.size(), csParam);

      init = true;
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
    }

    // need to reset x_sloppy every solve
    if (!param.use_sloppy_partial_accumulator) create_alias(x_sloppy, x);
  }

  // void BlockCG::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  // {
  //   if (param.is_preconditioner) commGlobalReductionPush(param.global_reduction);

  //   if (param.maxiter == 0 || param.Nsteps == 0) {
  //     if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
  //     return;
  //   }

  //   const int Np = 1;
  //   if (Np < 0 || Np > 16) errorQuda("Invalid value %d for solution_accumulator_pipeline", Np);

  //   // Determine whether or not we're doing a heavy quark residual
  //   const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

  //   // This check is pointless in the current version of the code, but it's being proactively added
  //   // just in case HQ residual solves are split into a separate file
  //   if (use_heavy_quark_res) errorQuda("The \"vanilla\" CG solver does not support HQ residual solves");

  //   /**
  //     When CG is used as a preconditioner, and we disable the `advanced features`, these features are turned off:
  //     - Reliable updates
  //     - Pipelining
  //     - Always use zero as the initial guess
  //     - Heavy quark residual
  //   */
  //   bool advanced_feature = !(param.precondition_no_advanced_feature && param.is_preconditioner);

  //   if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_INIT);

  //   // whether to select alternative reliable updates
  //   bool alternative_reliable = param.use_alternative_reliable;

  //   auto b2 = blas::norm2(b);

  //   // Check to see that we're not trying to invert on a zero-field source
  //   if (is_zero_src(x, b, b2)) {
  //     getProfile().TPSTOP(QUDA_PROFILE_INIT);
  //     return;
  //   }

  //   create(x, b);

  //   if (param.deflate) {
  //     // Construct the eigensolver and deflation space if requested.
  //     constructDeflationSpace(b[0], matEig);
  //     if (deflate_compute) {
  //       // compute the deflation space.
  //       if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_INIT);
  //       (*eig_solve)(evecs, evals);
  //       if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_INIT);
  //       deflate_compute = false;
  //     }
  //     if (recompute_evals) {
  //       eig_solve->computeEvals(evecs, evals);
  //       recompute_evals = false;
  //     }
  //   }

  //   const double u = precisionEpsilon(param.precision_sloppy);
  //   const double uhigh = precisionEpsilon(); // solver precision

  //   double Anorm = 0.0;
  //   vector<double> beta(b.size(), 0.0);

  //   // for alternative reliable updates
  //   if (advanced_feature && alternative_reliable) {
  //     // estimate norm for reliable updates
  //     mat(r[0], b[0]);
  //     Anorm = sqrt(blas::norm2(r[0]) / b2[0]);
  //   }

  //   // compute initial residual
  //   vector<double> r2(b2.size(), 0.0);
  //   if (advanced_feature && param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
  //     // Compute r = b - A * x
  //     mat(r, x);
  //     r2 = blas::xmyNorm(b, r);
  //     for (auto i = 0u; i < b.size(); i++)
  //       if (b2[i] == 0) b2[i] = r2[i];
  //     // y contains the original guess.
  //     blas::copy(y, x);
  //   } else {
  //     blas::copy(r, b);
  //     r2 = b2;
  //     blas::zero(y);
  //   }

  //   if (param.deflate && param.maxiter > 1) {
  //     // Deflate and accumulate to solution vector
  //     eig_solve->deflate(y, r, evecs, evals, true);
  //     mat(r, y);
  //     r2 = blas::xmyNorm(b, r);
  //   }

  //   MatrixXcd r2_mat(r2.size(), r2.size());
  //   for (int i = 0; i < r2.size(); i++) {
  //     r2_mat(i, i) = r2[i];
  //     for (int j = i + 1; j < r2.size(); j++) {
  //       r2_mat(i, j) = blas::cDotProduct(r[i], r[j]);
  //       r2_mat(j, i) = std::conj(r2_mat(i, j));
  //     }
  //   }

  //   blas::zero(x);
  //   if (param.use_sloppy_partial_accumulator) blas::zero(x_sloppy);
  //   blas::copy(r_sloppy, r);
  //   blas::copy(p, r_sloppy);
  //   blas::copy(rnew, r_sloppy);

  //   vector<double> r2_old(b.size(), 0.0);
  //   vector<double> alpha(b.size(), 0.0);
  //   MatrixXcd r2_old_mat(b.size(), b.size(), 0.0);
  //   MatrixXcd alpha_mat = MatrixXcd::Zero(b.size(), b.size());
  //   MatrixXcd beta_mat = MatrixXcd::Zero(b.size(), b.size());
  //   MatrixXcd pAp_mat = MatrixXcd::Identity(b.size(), b.size());
  //   MatrixXcd S = MatrixXcd::Identity(b.size(), b.size());
  //   MatrixXcd L = r2_mat.llt().matrixL();
  //   MatrixXcd C = L.adjoint();
  //   MatrixXcd Linv = C.inverse();
  //   vector<Complex> AC(b.size() * b.size());

  //   if (!param.is_preconditioner) {
  //     getProfile().TPSTOP(QUDA_PROFILE_INIT);
  //     getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);
  //   }

  //   auto stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

  //   vector<double> pAp(b.size());

  //   if (!param.is_preconditioner) {
  //     getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
  //     getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
  //   }

  //   int k = 0;

  //   PrintStats("CG", k, r2, b2);

  //   bool converged = convergenceL2(r2, stop);

  //   ReliableUpdatesParams ru_params;

  //   ru_params.alternative_reliable = alternative_reliable;
  //   ru_params.u = u;
  //   ru_params.uhigh = uhigh; // solver precision
  //   ru_params.Anorm = Anorm;
  //   ru_params.delta = param.delta;

  //   ru_params.maxResIncrease = param.max_res_increase;
  //   ru_params.maxResIncreaseTotal = param.max_res_increase_total;
  //   ru_params.use_heavy_quark_res = false; // since we've removed HQ residual support

  //   ReliableUpdates ru(ru_params, r2[0]);

  //   // set p to QR decompsition of r
  //   // temporary hack - use AC to pass matrix arguments to multiblas
  //   blas::zero(p);
  //   for (int i = 0; i < b.size(); i++) {
  //     for (int j = 0; j < b.size(); j++) {
  //       AC[i * b.size() + j] = Linv(i, j);
  //     }
  //   }
  //   blas::block::caxpy(AC, r, p);

  //   // set rsloppy to to QR decompoistion of r (p)
  //   blas::copy(r_sloppy, p);


  //   while (!converged && k < param.maxiter) {
  //     matSloppy(Ap, p);

  //     vector<double> sigma(b.size());

  //     bool breakdown = false;
  //     if (advanced_feature && param.pipeline) {
  //       errorQuda("pipeline not implemented");
  //     } else {
  //       r2_old = r2;

  //       // alternative reliable updates,
  //       if (advanced_feature && alternative_reliable) {
  //         auto pAppp = blas::cDotProductNormA(p, Ap);
  //         for (auto i = 0u; i < b.size(); i++) pAp[i] = pAppp[i].x;
  //         ru.update_ppnorm(pAppp[0].z); // using 0th system for RU
  //       } else {
  //         pAp = blas::reDotProduct(p, Ap);
  //       }

  //       for (int i = 0; i < b.size(); i++) {
  //         pAp_mat(i, i) = pAp[i];
  //         for (int j = i + 1; j < b.size(); j++) {
  //           pAp_mat(i, j) = blas::cDotProduct(p[i], Ap[j]);
  //           pAp_mat(j, i) = std::conj(pAp_mat(i, j));
  //         }
  //       }

  //       alpha_mat = pAp_mat.inverse() * C;
  //       for (int i = 0; i < b.size(); i++) {
  //         for (int j = 0; j < b.size(); j++) {
  //           AC[i * b.size() + j] = alpha_mat(i, j);
  //         }
  //       }
  //       blas::block::caxpy(AC, p, x_sloppy);

  //       beta_mat = pAp_mat.inverse() * C;
  //       for (int i = 0; i < b.size(); i++) {
  //         for (int j = 0; j < b.size(); j++) {
  //           AC[i * b.size() + j] = -beta_mat(i, j);
  //         }
  //       }
  //       blas::block::caxpy(AC, Ap, r_sloppy);

  //       for (auto i = 0u; i < b.size(); i++) alpha[i] = r2[i] / pAp[i];

  //       // here we are deploying the alternative beta computation
  //       auto cg_norm = blas::axpyCGNorm(-alpha, Ap, r_sloppy);
  //       for (auto i = 0u; i < b.size(); i++) {
  //         r2[i] = cg_norm[i].x;                                  // (r_new, r_new)
  //         sigma[i] = cg_norm[i].y >= 0.0 ? cg_norm[i].y : r2[i]; // use r2 if (r_k+1, r_k+1-r_k) breaks
  //       }

  //       blas::copy(rnew, r_sloppy);
  //       for (int i = 0; i < r2.size(); i++) {
  //         r2_mat(i, i) = r2[i];
  //         for (int j = i + 1; j < r2.size(); j++) {
  //           r2_mat(i, j) = blas::cDotProduct(r_sloppy[i], r_sloppy[j]);
  //           r2_mat(j, i) = std::conj(r2_mat(i, j));
  //         }
  //       }
  //       L = r2_mat.llt().matrixL();
  //       S = L.adjoint();
  //       Linv = S.inverse();
  //       blas::zero(r_sloppy);
  //       for (int i = 0; i < b.size(); i++) {
  //         for (int j = 0; j < b.size(); j++) {
  //           AC[i * b.size() + j] = Linv(i, j);
  //         }
  //       }
  //       blas::block::caxpy(AC, rnew, r_sloppy);

  //     }

  //     // reliable update conditions
  //     ru.update_rNorm(sqrt(r2[0]));

  //     if (advanced_feature) {
  //       ru.evaluate(r2_old[0]);
  //       // force a reliable update if we are within target tolerance (only if doing reliable updates)
  //       if (convergenceL2(r2, stop) && param.delta >= param.tol) ru.set_updateX();
  //     }

  //     if (!ru.trigger()) {
  //       for (auto i = 0u; i < beta.size(); i++) beta[i] = sigma[i] / r2_old[i]; // use the alternative beta computation

  //       // with Np=1 we just run regular fusion between x and p updates
  //       blas::axpyZpbx(alpha, p, x_sloppy, r_sloppy, beta);

  //       // alternative reliable updates
  //       if (advanced_feature) { ru.accumulate_norm(alpha[0]); }
  //     } else {

  //       for (auto i = 0u; i < b.size(); i++) {
  //         x_update_batch[i].accumulate_x(x_sloppy[i]);
  //         x_update_batch[i].reset_next();
  //       }
  //       blas::xpy(x_sloppy, y); // swap these around?

  //       mat(r, y); //  here we can use x as tmp
  //       r2 = blas::xmyNorm(b, r);

  //       if (param.deflate && sqrt(r2[0]) < ru.maxr_deflate * param.tol_restart) {
  //         // Deflate and accumulate to solution vector
  //         eig_solve->deflate(y, r, evecs, evals, true);

  //         // Compute r_defl = RHS - A * LHS
  //         mat(r, y);
  //         r2 = blas::xmyNorm(b, r);

  //         ru.update_maxr_deflate(r2[0]);
  //       }

  //       blas::copy(r_sloppy, r); // nop when these pointers alias
  //       blas::zero(x_sloppy);

  //       if (advanced_feature) { ru.update_norm(r2[0], y[0]); }

  //       if (advanced_feature) {
  //         // needed as a "dummy parameter" to reliable_break.
  //         bool L2breakdown = false;
  //         if (ru.reliable_break(r2[0], stop[0], L2breakdown, 0)) { break; }
  //       }

  //       auto rp = blas::cDotProduct(r_sloppy, p);
  //       for (auto i = 0u; i < b.size(); i++) rp[i] /= r2[i];
  //       blas::caxpy(-rp, r_sloppy, p);

  //       for (auto i = 0u; i < beta.size(); i++) beta[i] = r2[i] / r2_old[i];
  //       blas::xpayz(r_sloppy, beta, p, p_next);

  //       ru.reset(r2[0]);
  //     }

  //     breakdown = false;
  //     k++;

  //     PrintStats("CG", k, r2, b2);
  //     // check convergence
  //     converged = convergenceL2(r2, stop);

  //     // if we have converged and need to update any trailing solutions
  //     for (auto i = 0u; i < b.size(); i++) {
  //       if (converged && ru.steps_since_reliable > 0 && !x_update_batch[i].is_container_full()) {
  //         x_update_batch[i].accumulate_x(x_sloppy[i]);
  //       }

  //       if (ru.steps_since_reliable == 0) {
  //         x_update_batch[i].reset();
  //       } else {
  //         ++x_update_batch[i];
  //       }
  //     }
  //   }

  //   blas::copy(x, x_sloppy);
  //   blas::xpy(y, x);

  //   if (!param.is_preconditioner) {
  //     getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  //     getProfile().TPSTART(QUDA_PROFILE_EPILOGUE);

  //     param.iter += k;

  //     if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);
  //   }

  //   logQuda(QUDA_VERBOSE, "CG: Reliable updates = %d\n", ru.rUpdate);

  //   if (advanced_feature && param.compute_true_res) {
  //     // compute the true residuals
  //     mat(r, x);
  //     auto true_r2 = blas::xmyNorm(b, r);
  //     auto hq = blas::HeavyQuarkResidualNorm(x, r);
  //     for (auto i = 0u; i < b.size(); i++) {
  //       param.true_res[i] = sqrt(true_r2[i] / b2[i]);
  //       param.true_res_hq[i] = sqrt(hq[i].z);
  //     }
  //   }

  //   PrintSummary("CG", k, r2, b2, stop);

  //   if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);

  //   if (param.is_preconditioner) commGlobalReductionPop();
  // }

  cvector_ref<const ColorSpinorField> BlockCG::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    return r;
  }

  void BlockCG::blocksolve(ColorSpinorField &out, ColorSpinorField &in)
  {
    errorQuda("Not implemented");
  }

// #define MWVERBOSE 1

  void BlockCG::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    getProfile().TPSTART(QUDA_PROFILE_INIT);

    using Eigen::MatrixXcd;

    // Check to see that we're not trying to invert on a zero-field source
    // MW: it might be useful to check what to do here.
    double b2[QUDA_MAX_MULTI_SHIFT];
    double b2avg = 0;
    for (int i = 0; i < param.num_src; i++) {
      b2[i] = blas::norm2(b[i]);
      b2avg += b2[i];
      if (b2[i] == 0) {
        getProfile().TPSTOP(QUDA_PROFILE_INIT);
        errorQuda("Warning: inverting on zero-field source - undefined for block solver\n");
        blas::copy(x, b);
        param.true_res = 0.0;
        param.true_res_hq = 0.0;
        return;
      }
    }

    b2avg = b2avg / param.num_src;

    create(x, b);

    // calculate residuals for all vectors
    // and initialize r2 matrix
    double r2avg = 0;
    MatrixXcd r2(param.num_src, param.num_src);
    for (int i = 0; i < param.num_src; i++) {
      mat(r[i], x[i]);
      r2(i, i) = blas::xmyNorm(b[i], r[i]);
      r2avg += r2(i, i).real();
      printfQuda("r2[%i] %e\n", i, r2(i, i).real());
    }
    for (int i = 0; i < param.num_src; i++) {
      for (int j = i + 1; j < param.num_src; j++) {
        r2(i, j) = blas::cDotProduct(r[i], r[j]);
        r2(j, i) = std::conj(r2(i, j));
      }
    }

    blas::copy(r_sloppy, r);
    blas::copy(p, r_sloppy);
    blas::copy(rnew, r_sloppy);

    blas::copy(y, x);
    if (param.use_sloppy_partial_accumulator) blas::zero(x_sloppy);

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    if (use_heavy_quark_res) errorQuda("ERROR: heavy quark residual not supported in block solver");

    getProfile().TPSTOP(QUDA_PROFILE_INIT);
    getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

    double stop[QUDA_MAX_MULTI_SHIFT];

    for (int i = 0; i < param.num_src; i++) {
      stop[i] = stopping(param.tol, b2[i], param.residual_type); // stopping condition of solver
    }

    // Eigen Matrices instead of scalars
    MatrixXcd alpha = MatrixXcd::Zero(param.num_src, param.num_src);
    MatrixXcd beta = MatrixXcd::Zero(param.num_src, param.num_src);
    MatrixXcd C = MatrixXcd::Zero(param.num_src, param.num_src);
    MatrixXcd S = MatrixXcd::Identity(param.num_src, param.num_src);
    MatrixXcd pAp = MatrixXcd::Identity(param.num_src, param.num_src);
    vector<Complex> AC(param.num_src * param.num_src);

#ifdef MWVERBOSE
    MatrixXcd pTp = MatrixXcd::Identity(param.num_src, param.num_src);
#endif

    // FIXME:reliable updates currently not implemented
    /*
    double rNorm[QUDA_MAX_MULTI_SHIFT];
    double r0Norm[QUDA_MAX_MULTI_SHIFT];
    double maxrx[QUDA_MAX_MULTI_SHIFT];
    double maxrr[QUDA_MAX_MULTI_SHIFT];

    for(int i = 0; i < param.num_src; i++){
      rNorm[i] = sqrt(r2(i,i).real());
      r0Norm[i] = rNorm[i];
      maxrx[i] = rNorm[i];
      maxrr[i] = rNorm[i];
    }
    bool L2breakdown = false;
    int rUpdate = 0;
    nt steps_since_reliable = 1;
    */

    getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    int k = 0;

    PrintStats("CG", k, r2avg / param.num_src, b2avg);
    bool allconverged = true;
    bool converged[QUDA_MAX_MULTI_SHIFT];
    for (int i = 0; i < param.num_src; i++) {
      converged[i] = convergence(r2(i, i).real(), 0., stop[i], param.tol_hq);
      allconverged = allconverged && converged[i];
    }

    // CHolesky decomposition
    MatrixXcd L = r2.llt().matrixL(); //// retrieve factor L  in the decomposition
    C = L.adjoint();
    MatrixXcd Linv = C.inverse();

#ifdef MWVERBOSE
    std::cout << "r2\n " << r2 << std::endl;
    std::cout << "L\n " << L.adjoint() << std::endl;
#endif

    // set p to QR decompsition of r
    // temporary hack - use AC to pass matrix arguments to multiblas
    for (int i = 0; i < param.num_src; i++) {
      blas::zero(p[i]);
      for (int j = 0; j < param.num_src; j++) { AC[i * param.num_src + j] = Linv(i, j); }
    }
    blas::block::caxpy(AC, r, p);

    // set rsloppy to to QR decompoistion of r (p)
    for (int i = 0; i < param.num_src; i++) { blas::copy(r_sloppy[i], p[i]); }

#ifdef MWVERBOSE
    for (int i = 0; i < param.num_src; i++) {
      for (int j = 0; j < param.num_src; j++) { pTp(i, j) = blas::cDotProduct(p[i], p[j]); }
    }
    std::cout << " pTp  " << std::endl << pTp << std::endl;
    std::cout << " L " << std::endl << L.adjoint() << std::endl;
    std::cout << " C " << std::endl << C << std::endl;
#endif

    while (!allconverged && k < param.maxiter) {
      // apply matrix
      for (int i = 0; i < param.num_src; i++) {
        matSloppy(Ap[i], p[i]); // tmp as tmp
      }

      // calculate pAp
      for (int i = 0; i < param.num_src; i++) {
        for (int j = i; j < param.num_src; j++) {
          pAp(i, j) = blas::cDotProduct(p[i], Ap[j]);
          if (i != j) pAp(j, i) = std::conj(pAp(i, j));
        }
      }

      // update Xsloppy
      alpha = pAp.inverse() * C;
      // temporary hack using AC
      for (int i = 0; i < param.num_src; i++) {
        for (int j = 0; j < param.num_src; j++) { AC[i * param.num_src + j] = alpha(i, j); }
      }
      blas::block::caxpy(AC, p, x_sloppy);

      // update rSloppy
      beta = pAp.inverse();
      // temporary hack
      for (int i = 0; i < param.num_src; i++) {
        for (int j = 0; j < param.num_src; j++) { AC[i * param.num_src + j] = -beta(i, j); }
      }
      blas::block::caxpy(AC, Ap, r_sloppy);

      // orthorgonalize R
      // copy rSloppy to rnew as temporary
      for (int i = 0; i < param.num_src; i++) { blas::copy(rnew[i], r_sloppy[i]); }
      for (int i = 0; i < param.num_src; i++) {
        for (int j = i; j < param.num_src; j++) {
          r2(i, j) = blas::cDotProduct(r_sloppy[i], r_sloppy[j]);
          if (i != j) r2(j, i) = std::conj(r2(i, j));
        }
      }
      // Cholesky decomposition
      L = r2.llt().matrixL(); // retrieve factor L  in the decomposition
      S = L.adjoint();
      Linv = S.inverse();
      // temporary hack
      for (int i = 0; i < param.num_src; i++) {
        blas::zero(r_sloppy[i]);
        for (int j = 0; j < param.num_src; j++) { AC[i * param.num_src + j] = Linv(i, j); }
      }
      blas::block::caxpy(AC, rnew, r_sloppy);

#ifdef MWVERBOSE
      for (int i = 0; i < param.num_src; i++) {
        for (int j = 0; j < param.num_src; j++) {
          pTp(i, j) = blas::cDotProduct(r_sloppy[i], r_sloppy[j]);
        }
      }
      std::cout << " rTr " << std::endl << pTp << std::endl;
      std::cout << "QR" << S << std::endl << "QP " << S.inverse() * S << std::endl;
      ;
#endif

      // update p
      // use rnew as temporary again for summing up
      for (int i = 0; i < param.num_src; i++) { blas::copy(rnew[i], r_sloppy[i]); }
      // temporary hack
      for (int i = 0; i < param.num_src; i++) {
        for (int j = 0; j < param.num_src; j++) { AC[i * param.num_src + j] = std::conj(S(j, i)); }
      }
      blas::block::caxpy(AC, p, rnew);
      // set p = rnew
      for (int i = 0; i < param.num_src; i++) { blas::copy(p[i], rnew[i]); }

      // update C
      C = S * C;

#ifdef MWVERBOSE
      for (int i = 0; i < param.num_src; i++) {
        for (int j = 0; j < param.num_src; j++) { pTp(i, j) = blas::cDotProduct(p[i], p[j]); }
      }
      std::cout << " pTp " << std::endl << pTp << std::endl;
      std::cout << "S " << S << std::endl << "C " << C << std::endl;
#endif

      // calculate the residuals for all shifts
      r2avg = 0;
      for (int j = 0; j < param.num_src; j++) {
        r2(j, j) = C(0, j) * conj(C(0, j));
        for (int i = 1; i < param.num_src; i++) r2(j, j) += C(i, j) * conj(C(i, j));
        r2avg += r2(j, j).real();
      }

      k++;
      PrintStats("CG", k, r2avg / param.num_src, b2avg);
      // check convergence
      allconverged = true;
      for (int i = 0; i < param.num_src; i++) {
        converged[i] = convergence(r2(i, i).real(), 0.0, stop[i], param.tol_hq);
        allconverged = allconverged && converged[i];
      }
    }

    for (int i = 0; i < param.num_src; i++) { blas::xpy(y[i], x_sloppy[i]); }

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    getProfile().TPSTART(QUDA_PROFILE_EPILOGUE);

    param.iter += k;

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // logQuda(QUDA_VERBOSE, "CG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    for (int i = 0; i < param.num_src; i++) {
      mat(r[i], x[i]);
      param.true_res = sqrt(blas::xmyNorm(b[i], r[i]) / b2[i]);
      param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x[i], r[i]).z);
      param.true_res_offset[i] = param.true_res[0];
      param.true_res_hq_offset[i] = param.true_res_hq[0];

      PrintSummary("CG", k, r2(i, i).real(), b2[i], stop[i]);
    }

    getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);
    getProfile().TPSTART(QUDA_PROFILE_FREE);

    getProfile().TPSTOP(QUDA_PROFILE_FREE);

    return;
  }

} // namespace quda
