#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernels/gauge_utils.cuh>
#include <kernel.h>
#include <thread_local_cache.h>

namespace quda
{

  template <typename store_t, int nColor_, QudaReconstructType recon_, int stoutDim_>
  struct STOUTForceLambdaArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int stoutDim = stoutDim_;
    typedef typename gauge_mapper<store_t, recon>::type Gauge;
    typedef typename gauge_mapper<store_t, QUDA_RECONSTRUCT_NO>::type Force;

    Force lambda;
    const Force sigma;
    const Gauge in;

    int X[4]; // grid dimensions
    int border[4];
    const real rho;
    const int dir_ignore;
    const real anisotropy;

    STOUTForceLambdaArg(GaugeField &lambda, const GaugeField &sigma, const GaugeField &in, real rho, int dir_ignore,
                        real anisotropy) :
      kernel_param(dim3(1, 2, stoutDim)),
      lambda(lambda),
      sigma(sigma),
      in(in),
      rho(rho),
      dir_ignore(dir_ignore),
      anisotropy(anisotropy)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
        this->threads.x *= X[dir];
      }
      this->threads.x /= 2;
    }
  };

  template <typename Arg> struct STOUTForceLambda {

    const Arg &arg;
    constexpr STOUTForceLambda(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::real;
      using Link = Matrix<complex<real>, Arg::nColor>;

      // Compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }
      dir = dir + (dir >= arg.dir_ignore);

      Link U, Sigma, Stap, Q, USigma;

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, x, X, parity, dir, Stap, arg.dir_ignore, arg.anisotropy);

      // Get link U
      U = arg.in(dir, linkIndex(x, X), parity);
      Sigma = arg.sigma(dir, x_cb, parity);
      USigma = U * Sigma;

      // Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag
      //--------------------------------------------------------------------
      // Compute \Omega = \rho * S * U^{\dagger}
      Q = (arg.rho * Stap) * conj(U);
      // Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu}
      //                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]
      makeHerm(Q);
      // Q is now defined.

      Link Lambda = deriv_exponentiate_iQ(Q, USigma);
      makeRealHerm(Lambda);
      arg.lambda(dir, linkIndex(x, X), parity) = Lambda;
    }
  };

  template <typename store_t, int nColor_, QudaReconstructType recon_, int stoutDim_>
  struct STOUTForceSigmaArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int stoutDim = stoutDim_;
    typedef typename gauge_mapper<store_t, recon>::type Gauge;
    typedef typename gauge_mapper<store_t, QUDA_RECONSTRUCT_NO>::type Force;

    Force sigma;
    const Force lambda;
    const Gauge in;

    int X[4]; // grid dimensions
    int border[4];
    const real rho;
    const int dir_ignore;
    const real anisotropy;

    STOUTForceSigmaArg(GaugeField &sigma, const GaugeField &lambda, const GaugeField &in, real rho, int dir_ignore,
                       real anisotropy) :
      kernel_param(dim3(1, 2, stoutDim)),
      sigma(sigma),
      lambda(lambda),
      in(in),
      rho(rho),
      dir_ignore(dir_ignore),
      anisotropy(anisotropy)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
        this->threads.x *= X[dir];
      }
      this->threads.x /= 2;
    }
  };

  template <typename Arg> struct STOUTForceSigma {

    const Arg &arg;
    constexpr STOUTForceSigma(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::real;
      using Link = Matrix<complex<real>, Arg::nColor>;

      // Compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }
      dir = dir + (dir >= arg.dir_ignore);

      Link U, Sigma, Stap, Q, USigma;

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStaple(arg, x, X, parity, dir, Stap, arg.dir_ignore, arg.anisotropy);

      // Get link U
      U = arg.in(dir, linkIndex(x, X), parity);
      Sigma = arg.sigma(dir, x_cb, parity);
      Link Lambda = arg.lambda(dir, linkIndex(x, X), parity);
      const complex<real> i {0.0, 1.0};

      // Compute Omega_{mu}=[Sum_{mu neq nu}rho_{mu,nu}C_{mu,nu}]*U_{mu}^dag
      //--------------------------------------------------------------------
      // Compute \Omega = \rho * S * U^{\dagger}
      Q = (arg.rho * Stap) * conj(U);
      // Compute \Q_{mu} = i/2[Omega_{mu}^dag - Omega_{mu}
      //                      - 1/3 Tr(Omega_{mu}^dag - Omega_{mu})]
      makeHerm(Q);
      // Q is now defined.

      Link exp_iQ = exponentiate_iQ(Q);
      Sigma = Sigma * exp_iQ;
      Sigma += i * conj(arg.rho * Stap) * Lambda;

      USigma = Link();
      int mu = dir;
      packed_array<int8_t, 4> dx = {};
#pragma unroll
      for (int nu = 0; nu < 4; nu++) {
        // Identify directions orthogonal to the link and
        // ignore the dir_ignore direction (usually the temporal dim
        // when used with STOUT or APE for measurement smearing)

        if (nu != mu && nu != arg.dir_ignore) {
          {
            // Get link U_{\nu}(x)
            Link U1 = arg.in(nu, linkIndexShift(x, dx, X), parity);
            Link L1 = arg.lambda(nu, linkIndexShift(x, dx, X), parity);

            // Get link U_{\mu}(x+\nu)
            dx[nu]++;
            Link U2 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);
            Link L2 = arg.lambda(mu, linkIndexShift(x, dx, X), 1 - parity);
            dx[nu]--;

            // Get link U_{\nu}(x+\mu)
            dx[mu]++;
            Link U3 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);
            Link L3 = arg.lambda(nu, linkIndexShift(x, dx, X), 1 - parity);
            dx[mu]--;

            Link temp1 = U3 * conj(U2) * conj(U1) * L1;
            Link temp5 = L3 * U3 * conj(U2) * conj(U1);
            Link temp6 = U3 * conj(U2) * L2 * conj(U1);
            USigma += temp1 - temp5 + temp6;
          }

          {
            // Get link U_{\mu}(x-\mu)
            dx[nu]--;
            Link U1 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);
            Link L1 = arg.lambda(nu, linkIndexShift(x, dx, X), 1 - parity);
            // Get link U_{\nu}(x-\mu)
            Link U2 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);
            Link L2 = arg.lambda(mu, linkIndexShift(x, dx, X), 1 - parity);

            // Get link U_{\mu}(x-\mu+\nu)
            dx[mu]++;
            Link U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);
            Link L3 = arg.lambda(nu, linkIndexShift(x, dx, X), parity);

            // reset dx
            dx[mu]--;
            dx[nu]++;

            Link temp2 = conj(U3) * conj(U2) * L2 * U1;
            Link temp3 = conj(U3) * L3 * conj(U2) * U1;
            Link temp4 = conj(U3) * conj(U2) * L1 * U1;
            USigma += temp2 + temp3 - temp4;
          }
        }
      }

      Sigma -= i * arg.rho * USigma;

      arg.sigma(dir, x_cb, parity) = Sigma;
    }
  };

} // namespace quda
