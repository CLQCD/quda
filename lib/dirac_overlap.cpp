#include <util_quda.h>
#include <dirac_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>

namespace quda
{
  DiracOverlap::DiracOverlap(const DiracParam &param) : DiracWilson(param), overlap_kernel(param.overlap_kernel) { }

  DiracOverlap::DiracOverlap(const DiracOverlap &dirac) : DiracWilson(dirac), overlap_kernel(dirac.overlap_kernel) { }

  DiracOverlap::~DiracOverlap() { }

  DiracOverlap &DiracOverlap::operator=(const DiracOverlap &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
      overlap_kernel = dirac.overlap_kernel;
    }
    return *this;
  }

#define flip(x) (x) = ((x) == QUDA_DAG_YES ? QUDA_DAG_NO : QUDA_DAG_YES)

  // Defined as 0.5 * (1 + \gamma_5 sign(\gamma_5 W)) where W is the Wilson M operator
  void DiracOverlap::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, QudaParity) const
  {
    auto b1 = getFieldTmp(out);
    auto b2 = getFieldTmp(out);
    auto Mb1 = getFieldTmp(out);
    auto Ab1 = getFieldTmp(out);
    auto deflated = getFieldTmp(out);

    cvector_ref<ColorSpinorField> &evecs = overlap_kernel->evecs;
    cvector<double> &evals = overlap_kernel->evals;
    const size_t size = overlap_kernel->Size();
    const double lambda_max = (1.0 + 8.0 * kappa);
    const double evals_max = evals[size - 1];
    const double epsilon = evals_max * evals_max;

    const bool dagger_yes = (dagger == QUDA_DAG_YES);

    if (dagger_yes) {
      gamma5(deflated, in);
    } else {
      blas::copy(deflated, in);
    }

    /**
     * \gamma_5 * sign(\gamma_5 M) for small eigenvalues
     * Define the eigenvalues and eigenvectors \gamma_5 M v_i = \lambda_i v_i
     * \gamma_5 \sum_i sign(\lambda_i) |v_i><v_i|
     */
    std::vector<Complex> s(size);
    blas::block::cDotProduct(s, evecs, deflated);
    for (int i = 0; i < size; i++) { s[i] *= -1; }
    blas::block::caxpy(s, evecs, deflated);
    for (int i = 0; i < size; i++) { s[i] *= -evals[i] / abs(evals[i]); }
    for (auto &field : out) { field.get().zero(); }
    blas::block::caxpy(s, evecs, out);
    if (!dagger_yes) { gamma5(out, out); }

    /**
     * \gamma_5 * sign(\gamma_5 M) for large eigenvalues
     * Define the Chebyshev polynomial approximation P(x) ~ x^{-1/2}
     * M P(M^\dagger M)
     * An extra factor of \lambda_{max} is included in the definition of P(x)
     */
    blas::zero(b1);
    blas::zero(b2);
    if (dagger_yes) { flip(dagger); } // Make sure we are performing P(M^\dagger M)
    for (int k = overlap_kernel->remez_order[0]; k >= 0; --k) {
      DiracWilson::M(Mb1, b1);
      flip(dagger);
      DiracWilson::M(Ab1, Mb1);
      flip(dagger);
      blas::axpby(-(1 + epsilon) / (1 - epsilon), b1, 2 / (1 - epsilon) / (lambda_max * lambda_max), Ab1);
      if (k > 0) {
        blas::axpbypczw(overlap_kernel->remez_coeff[0][k], deflated, 2.0, Ab1, -1.0, b2, b2);
      } else {
        blas::axpbypczw(overlap_kernel->remez_coeff[0][0], deflated, 1.0, Ab1, -1.0, b2, b2);
      }
      std::swap(b1, b2);
    }
    DiracWilson::M(Mb1, b1);
    if (dagger_yes) { flip(dagger); }
    if (dagger_yes) { gamma5(Mb1, Mb1); }

    /**
     * 0.5 * (1 + \gamma_5 sign(\gamma_5 M))
     */
    blas::axpbypczw(0.5, in, 0.5 / lambda_max, Mb1, 0.5, out, out);
  }

  void DiracOverlap::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                QudaParity parity, cvector_ref<const ColorSpinorField> &x, double scale) const
  {
    Dslash(out, in, parity);
    if (scale != 0.0) { blas::axpy(scale, x, out); }
  }

  // Defined as m / (2\rho - m) + D
  void DiracOverlap::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    const double two_rho = 8.0 - 1.0 / kappa;
    DslashXpay(out, in, QUDA_INVALID_PARITY, in, mass / (two_rho - mass));
  }

  // Defined as m^2 / (2\rho^2 - m^2) + DdagD
  void DiracOverlap::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    const double two_rho = 8.0 - 1.0 / kappa;
    auto tmp = getFieldTmp(out);
    Dslash(tmp, in, QUDA_INVALID_PARITY);
    DslashXpay(out, tmp, QUDA_INVALID_PARITY, in, (mass * mass) / (two_rho * two_rho - mass * mass));
  }

  // Defined as m^2 / (2\rho^2 - m^2) + DdagD
  // (1\pm\gamma_5)/2 DdagD (1\pm\gamma_5)/2 = (1\pm\gamma_5)/2 D (1\pm\gamma_5)/2
  void DiracOverlap::MdagMChiral(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    const double two_rho = 8.0 - 1.0 / kappa;
    DslashXpay(out, in, QUDA_INVALID_PARITY, in, (mass * mass) / (two_rho * two_rho - mass * mass));
  }

  void DiracOverlap::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                             cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                             const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    for (auto i = 0u; i < b.size(); i++) {
      src[i] = const_cast<ColorSpinorField &>(b[i]).create_alias();
      sol[i] = x[i].create_alias();
    }
  }

  void DiracOverlap::reconstruct(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                 const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) { return; }
    auto tmp = getFieldTmp(out);
    const double two_rho = 8.0 - 1.0 / kappa;

    if (solType == QUDA_MAT_SOLUTION) {
      // 1 / (2\rho - m)
      blas::axy(1.0 / (two_rho - mass), out, tmp);
    } else if (solType == QUDA_MATDAG_MAT_SOLUTION) {
      // m / (2\rho - m) + D^dagger
      Mdag(tmp, out);
      // 1 / (2\rho + m)
      blas::ax(1.0 / (two_rho + mass), tmp);
    }
    // 1 - D
    blas::axpbyz(-1.0 / (two_rho - mass), in, two_rho / (two_rho - mass), tmp, out);
  }

  void DiracOverlap::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
  }
} // namespace quda