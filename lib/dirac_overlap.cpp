#include "util_quda.h"
#include <dirac_quda.h>
// #include <dslash_quda.h>
#include <blas_quda.h>
#include <multigrid.h>
#include <eigensolve_quda.h>
#include <gauge_field.h>

namespace quda
{
  // Chebyshev polynomial the first kind
  // T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x)
  double Tn(double x, int n)
  {
    if (abs(x) <= 1.0) { return cos(n * acos(x)); }
    double T0 = 1, T1 = x, Tk;
    switch (n) {
    case 0: return T0;
    case 1: return T1;
    default:
      for (int k = 2; k <= n; ++k) {
        Tk = 2 * x * T1 - T0;
        T0 = T1;
        T1 = Tk;
      }
      return Tk;
    }
  }

  // \sum_{i=0}^n c_i T_i
  // T_{k+1}(x) = 2 x T_k(x) - T_{k-1}(x)
  // Use Clenshaw algorithm
  double ciTi(double x, std::vector<double> c, int n)
  {
    double b2 = 0.0, b1 = 0.0, bk;
    for (int k = n; k >= 1; --k) {
      bk = c[k] + 2 * x * b1 - b2;
      b2 = b1;
      b1 = bk;
    }
    return c[0] + x * b1 - b2;
  }

  // (\sum_{i=0}^n c_i T_i)' = \sum_{i=1}^n i c_i U_{i-1}
  // U_{k+1}(x) = 2 x U_k(x) - U_{k-1}
  // Use Clenshaw algorithm
  double iciUim1(double x, std::vector<double> c, int n)
  {
    double b2 = 0.0, b1 = 0.0, bk;
    for (int k = n - 1; k >= 1; --k) {
      bk = (k + 1) * c[k + 1] + 2 * x * b1 - b2;
      b2 = b1;
      b1 = bk;
    }
    return c[1] + 2 * x * b1 - b2;
  }

  double residual(double x, std::vector<double> c, int n, double epsilon, bool derivative)
  {
    const double z = (x * 2 - (1 + epsilon)) / (1 - epsilon);
    if (derivative) {
      return -1 / (2 * sqrt(x)) * ciTi(z, c, n) - sqrt(x) * iciUim1(z, c, n) * (2 / (1 - epsilon));
    } else {
      return 1 - sqrt(x) * ciTi(z, c, n);
    }
  }

  double find_root(double x_l, double x_r, std::vector<double> c, int n, double epsilon, bool derivative)
  {
    double x_m, res_r, res_l, res_m;

    res_l = residual(x_l, c, n, epsilon, derivative);
    res_r = residual(x_r, c, n, epsilon, derivative);
    if (abs(res_l) < 1e-15) return x_l;
    if (abs(res_r) < 1e-15) return x_r;
    if (res_r * res_l > 0)
      printf("ERROR: find_root with derivative=%d called with wrong ends: (%e %e)->(%e %e)\n", derivative, x_l, x_r,
             res_l, res_r);
    for (int i = 0; i < 10; i++) {
      x_m = (res_l * x_r - res_r * x_l) / (res_l - res_r);
      res_m = residual(x_m, c, n, epsilon, derivative);
      if (res_m * res_l > 0) {
        x_l = x_m;
        res_l = res_m;
      } else {
        x_r = x_m;
        res_r = res_m;
      }
    }
    return (res_l * x_r - res_r * x_l) / (res_l - res_r);
  }

  std::vector<double> minimaxApproximationRemez(double delta, double epsilon)
  {
    const int n = ceil(-log(delta / 0.41) / (2.083 * sqrt(epsilon))) + 1;
    constexpr int max_iter = 5;
    std::vector<double> y(n + 1), z(n + 1), c(n + 1), b(n + 1);
    Eigen::Map<Eigen::VectorXd> b_eigen(b.data(), b.size()), c_eigen(c.data(), c.size());
    Eigen::MatrixXd M_eigen(n + 1, n + 1);

    for (int i = 0; i < n + 1; ++i) {
      z[i] = cos(M_PI * i / n);
      y[i] = (z[i] * (1 - epsilon) + (1 + epsilon)) / 2;
    }

    for (int iter = 0; iter < max_iter; ++iter) {
      // Construct matrix M_ij=\sqrt{y_i}T_j(z_i)
      for (int i = 0; i < n + 1; ++i) {
        for (int j = 0; j < n; ++j) { M_eigen(i, j) = sqrt(y[i]) * Tn(z[i], j); }
        M_eigen(i, n) = i % 2 == 0 ? 1 : -1; // T_n is not a real Chebyshev polynomial
        b_eigen(i) = 1.0;
      }
      c_eigen = M_eigen.lu().solve(b_eigen);

      // Drop T_n
      for (int i = 0; i < n; ++i) { b[i] = find_root(y[i], y[i + 1], c, n - 1, epsilon, false); }
      for (int i = n - 1; i > 0; --i) { y[i] = find_root(b[i], b[i - 1], c, n - 1, epsilon, true); }
      for (int i = 1; i < n; ++i) { z[i] = (2 * y[i] - (1 + epsilon)) / (1 - epsilon); }
      for (int i = 0; i < n + 1; ++i) { b[i] = abs(1 - sqrt(y[i]) * ciTi(z[i], c, n - 1)); }
      if (*std::max_element(b.begin(), b.end()) <= delta) { return {c.begin(), c.begin() + n}; }
    }
    errorQuda("minimaxApproximationRemez can not converge");
  }

  DiracOverlap::DiracOverlap(const DiracParam &param) :
    DiracWilson(param), mass_overlap(param.mass), wilson(new DiracWilson(param)), zero_shift(0.0), M_mass(false)
  {
  }

  DiracOverlap::DiracOverlap(const DiracOverlap &dirac) :
    DiracWilson(dirac),
    mass_overlap(dirac.mass_overlap),
    wilson(dirac.wilson),
    hermitian_wilson_n_eig(dirac.hermitian_wilson_n_eig),
    hermitian_wilson_evecs(dirac.hermitian_wilson_evecs),
    hermitian_wilson_evals(dirac.hermitian_wilson_evals),
    remez_tol(dirac.remez_tol),
    remez_n(dirac.remez_n),
    remez_c(dirac.remez_c),
    zero_shift(dirac.zero_shift),
    M_mass(dirac.M_mass)
  {
  }

  DiracOverlap::~DiracOverlap() { delete wilson; }

  DiracOverlap &DiracOverlap::operator=(const DiracOverlap &dirac)
  {
    if (&dirac != this) {
      DiracWilson::operator=(dirac);
      mass_overlap = dirac.mass_overlap;
      wilson = dirac.wilson;
      hermitian_wilson_n_eig = dirac.hermitian_wilson_n_eig;
      hermitian_wilson_evecs = dirac.hermitian_wilson_evecs;
      hermitian_wilson_evals = dirac.hermitian_wilson_evals;
      remez_tol = dirac.remez_tol;
      remez_n = dirac.remez_n;
      remez_c = dirac.remez_c;
      zero_shift = dirac.zero_shift;
      M_mass = dirac.M_mass;
    }
    return *this;
  }

#define flip(x) (x) = ((x) == QUDA_DAG_YES ? QUDA_DAG_NO : QUDA_DAG_YES)

  void DiracOverlap::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    printfQuda("Entering DiracOverlap::M\n");

    auto tmp1 = getFieldTmp(in[0]);
    auto tmp2 = getFieldTmp(in[0]);
    auto tmp3 = getFieldTmp(in[0]);
    auto tmp4 = getFieldTmp(in[0]);
    auto tmp5 = getFieldTmp(in[0]);
    ColorSpinorField &b1 = tmp1;
    ColorSpinorField &b2 = tmp2;
    ColorSpinorField &Mb1 = tmp3;
    ColorSpinorField &Ab1 = tmp4;
    ColorSpinorField &deflated = tmp5;

    const double hermitian_wilson_evals_max = hermitian_wilson_evals[hermitian_wilson_n_eig - 1];
    const double epsilon = hermitian_wilson_evals_max * hermitian_wilson_evals_max;
    const double lambda_max = (1 + 8 * kappa);
    const double rho = 4 - 1 / (2 * kappa);

    // signLow(out, deflated, hermitian_wilson_evecs, hermitian_wilson_evals, hermitian_wilson_n_eig);
    std::vector<Complex> s(hermitian_wilson_n_eig);
    blas::block::cDotProduct(s, hermitian_wilson_evecs, in);
    for (int i = 0; i < hermitian_wilson_n_eig; i++) { s[i] *= -1; }
    blas::block::caxpyz(s, hermitian_wilson_evecs, in, deflated);
    for (int i = 0; i < hermitian_wilson_n_eig; i++) {
      s[i] *= -hermitian_wilson_evals[i] / abs(hermitian_wilson_evals[i]);
    }
    for (auto &field : out) { field.get().zero(); }

    blas::block::caxpy(s, hermitian_wilson_evecs, out);

    gamma5(out, out);

    // signHighPolynomial(b1, b2, Ab1, deflated, mat, remez_c, remez_n, epsilon, lambda_max);
    b1.zero();
    b2.zero();
    for (int k = remez_n; k >= 1; --k) {
      DiracWilson::M(Mb1, b1);
      flip(dagger);
      DiracWilson::M(Ab1, Mb1);
      flip(dagger);
      blas::axpby(-(1 + epsilon) / (1 - epsilon), b1, 2 / (1 - epsilon) / (lambda_max * lambda_max), Ab1);
      blas::axpbypczw(remez_c[k], deflated, 2.0, Ab1, -1.0, b2, b2);
      std::swap(b1, b2);
    }

    DiracWilson::M(Mb1, b1);
    flip(dagger);
    DiracWilson::M(Ab1, Mb1);
    flip(dagger);
    blas::axpby(-(1 + epsilon) / (1 - epsilon), b1, 2 / (1 - epsilon) / (lambda_max * lambda_max), Ab1);
    blas::axpbypczw(remez_c[0], deflated, 1.0, Ab1, -1.0, b2, b2);
    DiracWilson::M(b1, b2);
    blas::axpbypczw(rho, in, rho / lambda_max, b1, rho, out, out);

    if (M_mass) {
      // out = ((1 - m / (2 * \rho)) * D_{ov} + m) * in
      printfQuda("=====M_mass=====\n");
      double alpha = 1.0 - (mass_overlap / (2.0 * rho));
      double beta = mass_overlap;
      blas::axpbyz(alpha, out, beta, in, out);
    }

    if (zero_shift != 0.0) {
      printfQuda("=====zero_shift overlap=====\n");
      blas::axpby(zero_shift, in, 1.0, out);
    }
  }

  void DiracOverlap::Mdag(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    printfQuda("Entering DiracOverlap::Mdag\n");
    checkFullSpinor(out, in);
    auto tmp_1 = getFieldTmp(out);
    auto tmp_2 = getFieldTmp(out);

    // M^\dag*x = \gamma_5*M*\gamma_5*x
    gamma5(tmp_1, in);
    M(tmp_2, tmp_1);
    gamma5(out, tmp_2);
  }

  void DiracOverlap::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    printfQuda("Entering DiracOverlap::MdagM\n");
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracOverlap::MMdag(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    printfQuda("Entering DiracOverlap::MMdag\n");
    auto tmp = getFieldTmp(out);
    Mdag(tmp, in);
    M(out, tmp);
  }

  // void DiracOverlap::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  // {
  //   printfQuda("Entering DiracOverlap::MdagM\n");
  //   checkFullSpinor(out, in);
  //   auto tmp_1 = getFieldTmp(in[0]);
  //   auto tmp_2 = getFieldTmp(in[0]);

  //   // Mdag*M*x = \gamma_5*M*\gamma_5*M*x
  //   M(tmp_1, in);
  //   gamma5(tmp_1, tmp_1);
  //   M(tmp_2, tmp_1);
  //   gamma5(out, tmp_2);
  // }

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

  void DiracOverlap::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                 const QudaSolutionType) const
  {
    // do nothing
    printfQuda("DiracOverlap::reconstruct not implemented!\n");
  }

  void DiracOverlap::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
  }

  void DiracOverlap::setupHermitianWilson(int n_eig, const std::vector<ColorSpinorField> &evecs,
                                          const std::vector<Complex> &evals, double invsqrt_tol) const
  {
    printfQuda("Entering DiracOverlap::setupHermitianWilson\n");
    hermitian_wilson_n_eig = n_eig;
    hermitian_wilson_evecs.resize(n_eig);
    hermitian_wilson_evals.resize(n_eig);

    const double lambda_max = 1 + 8 * kappa;
    ColorSpinorParam cudaParam(evecs[0]);
    if (evecs[0].Precision() == gauge->Precision()) {
      cudaParam.create = QUDA_REFERENCE_FIELD_CREATE;
      for (int i = 0; i < n_eig; i++) {
        cudaParam.v = evecs[i].data();
        hermitian_wilson_evecs[i] = ColorSpinorField(cudaParam);
        hermitian_wilson_evals[i] = evals[i].real() / lambda_max;
      }
    } else {
      cudaParam.create = QUDA_NULL_FIELD_CREATE;
      cudaParam.setPrecision(gauge->Precision(), gauge->Precision(), true);
      for (int i = 0; i < n_eig; i++) {
        hermitian_wilson_evecs[i] = ColorSpinorField(cudaParam);
        hermitian_wilson_evecs[i] = evecs[i];
        hermitian_wilson_evals[i] = evals[i].real() / lambda_max;
      }
    }
    remez_tol = invsqrt_tol;
    remez_c = minimaxApproximationRemez(invsqrt_tol, pow(hermitian_wilson_evals[n_eig - 1], 2));
    remez_n = remez_c.size() - 1;
  }

} // namespace quda