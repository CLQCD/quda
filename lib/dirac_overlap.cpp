#include <util_quda.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <blas_quda.h>

namespace quda
{
  DiracOverlap::DiracOverlap(const DiracParam &param) : Dirac(param), overlap_kernel(param.overlap_kernel) { }

  DiracOverlap::DiracOverlap(const DiracOverlap &dirac) : Dirac(dirac), overlap_kernel(dirac.overlap_kernel) { }

  DiracOverlap::~DiracOverlap() { }

  DiracOverlap &DiracOverlap::operator=(const DiracOverlap &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
      overlap_kernel = dirac.overlap_kernel;
    }
    return *this;
  }

  /**
   * Defined as 0.5 * (1 + \gamma_5 sign(\gamma_5 M)) where M is the Wilson operator
   */
  void DiracOverlap::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, QudaParity parity) const
  {
    auto in_def = getFieldTmp(out);
    auto b1 = getFieldTmp(out);
    auto b2 = getFieldTmp(out);
    auto Mb1 = getFieldTmp(out);
    auto Ab1 = getFieldTmp(out);

    cvector_ref<ColorSpinorField> &evecs = overlap_kernel->evecs;
    cvector<double> &evals = overlap_kernel->evals;
    const double remez_order = overlap_kernel->remez_order[0];
    cvector<double> &remez_coeff = overlap_kernel->remez_coeff[0];
    const double lambda_max = (1.0 + 8.0 * overlap_kernel->kappa);
    const double epsilon = overlap_kernel->epsilon;

    const bool dagger_yes = (dagger == QUDA_DAG_YES);

    /**
     * Apply 0.5 directly to the input
     */
    if (dagger_yes) {
      blas::axy(0.5, in, out);
      gamma5(in_def, out);
    } else {
      blas::axy(0.5, in, in_def);
      gamma5(out, in_def);
    }

    /**
     * \gamma_5 sign(\gamma_5 M) for small eigenvalues
     * Define the eigenvalues and eigenvectors \gamma_5 M v_i = \lambda_i v_i
     * ==> \gamma_5 \sum_i sign(\lambda_i) |v_i><v_i|
     */
    std::vector<quda::Complex> alpha(evecs.size() * in_def.size());
    blas::block::cDotProduct(alpha, evecs, in_def);
    for (int i = 0; i < evecs.size() * in_def.size(); i++) { alpha[i] *= -1; }
    blas::block::caxpy(alpha, evecs, in_def);
    for (int i = 0; i < evecs.size(); i++) {
      for (int j = 0; j < in_def.size(); ++j) { alpha[i * in_def.size() + j] *= -evals[i] / abs(evals[i]); }
    }
    blas::block::caxpy(alpha, evecs, out);
    if (!dagger_yes) { gamma5(out, out); }

    /**
     * \gamma_5 sign(\gamma_5 M) for large eigenvalues
     * Define the Chebyshev polynomial approximation P(x) ~ x^{-1/2}
     * ==> M P(M^\dagger M)
     * Here M is the normalized Wilson operator which has the maximum eigenvalue 1
     */
    blas::zero(b1);
    blas::zero(b2);
    for (int k = remez_order; k >= 0; --k) {
      ApplyWilson(Mb1, b1, *gauge, -kappa, b1, parity, QUDA_DAG_NO, commDim.data, profile);
      ApplyWilson(Ab1, Mb1, *gauge, -kappa, Mb1, parity, QUDA_DAG_YES, commDim.data, profile);
      blas::axpby(-(1.0 + epsilon) / (1.0 - epsilon), b1, 2.0 / (1.0 - epsilon) / (lambda_max * lambda_max), Ab1);
      if (k > 0) {
        blas::axpbypczw(remez_coeff[k], in_def, 2.0, Ab1, -1.0, b2, b2);
      } else {
        blas::axpbypczw(remez_coeff[0], in_def, 1.0, Ab1, -1.0, b2, b2);
      }
      std::swap(b1, b2);
    }
    ApplyWilson(Mb1, b1, *gauge, -kappa, b1, parity, QUDA_DAG_NO, commDim.data, profile);
    if (dagger_yes) { gamma5(Mb1, Mb1); }

    /**
     * 0.5 * (1 + \gamma_5 sign(\gamma_5 M))
     */
    blas::axpy(1.0 / lambda_max, Mb1, out);
  }

  void DiracOverlap::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    Dslash(out, in, parity);
    if (k != 0.0) { blas::axpy(k, x, out); }
  }

  // Defined as m / (2\rho - m) + D
  void DiracOverlap::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    printfQuda("Entering DiracOverlap::M\n");
    const double two_rho = 8.0 - 1.0 / kappa;
    DslashXpay(out, in, QUDA_INVALID_PARITY, in, mass / (two_rho - mass));
  }

  // Defined as m^2 / (2\rho^2 - m^2) + DdagD
  void DiracOverlap::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    printfQuda("Entering DiracOverlap::MdagM\n");
    const double two_rho = 8.0 - 1.0 / kappa;
    auto tmp = getFieldTmp(out);
    Dslash(tmp, in, QUDA_INVALID_PARITY);
    DslashXpay(out, tmp, QUDA_INVALID_PARITY, in, (mass * mass) / (two_rho * two_rho - mass * mass));
  }

  // Defined as m^2 / (2\rho^2 - m^2) + D
  // (1\pm\gamma_5)/2 DdagD (1\pm\gamma_5)/2 = (1\pm\gamma_5)/2 D (1\pm\gamma_5)/2
  void DiracOverlap::MdagMChiral(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, QudaChirality chirality) const
  {
    printfQuda("Entering DiracOverlap::MdagMChiral\n");
    const double two_rho = 8.0 - 1.0 / kappa;
    ColorSpinorParam param(in[0]);
    param.create = QUDA_NULL_FIELD_CREATE;
    param.nSpin = 4;
    param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    auto in_tmp = getFieldTmp<ColorSpinorField>(in.size(), param);
    auto out_tmp = getFieldTmp<ColorSpinorField>(out.size(), param);

    spinorChiralEmbed(in_tmp[0], in[0], chirality);
    DslashXpay(out_tmp, in_tmp, QUDA_INVALID_PARITY, in_tmp, (mass * mass) / (two_rho * two_rho - mass * mass));
    spinorChiralProject(out[0], out_tmp[0], chirality);
  }

  void DiracOverlap::prepare(cvector_ref<ColorSpinorField> &out, cvector_ref<ColorSpinorField> &in,
                             cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                             const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    for (auto i = 0u; i < b.size(); i++) {
      in[i] = const_cast<ColorSpinorField &>(b[i]).create_alias();
      out[i] = x[i].create_alias();
    }
  }

  void DiracOverlap::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                 const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) { return; }
    const double two_rho = 8.0 - 1.0 / kappa;

    if (solType == QUDA_MAT_SOLUTION) {
      // // 1 / (2\rho - m)
      // blas::ax(1.0 / (two_rho - mass), x);
      // // 1 - D
      // blas::axpby(-1.0 / (two_rho - mass), b, two_rho / (two_rho - mass), x);

      // auto tmp = getFieldTmp(x);
      // // m / (2\rho - m) + D^dagger
      // Mdag(tmp, x);
      // 1 / (2\rho + m)
      blas::ax(1.0 / (two_rho + mass), x);
      // 1 - D
      blas::axpby(-1.0 / (two_rho - mass), b, two_rho / (two_rho - mass), x);
    } else if (solType == QUDA_MATDAG_MAT_SOLUTION) {
      // 1 / (4\rho^2 - m^2)
      blas::ax(1.0 / (two_rho * two_rho - mass * mass), x);
      // 1 - D^\dagger D
      blas::axpby(-1.0 / (two_rho * two_rho - mass * mass), b, two_rho * two_rho / (two_rho * two_rho - mass * mass), x);
    }
  }

  void DiracOverlap::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
  }
} // namespace quda