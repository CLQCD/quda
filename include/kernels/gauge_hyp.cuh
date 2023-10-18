#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <quda_matrix.h>
#include <su3_project.cuh>
#include <kernel.h>
#include <kernels/gauge_utils.cuh>

namespace quda
{

#define  DOUBLE_TOL	1e-15
#define  SINGLE_TOL	2e-6

  template <typename Float_, int nColor_, QudaReconstructType recon_, int apeDim_>
  struct GaugeHYPArg : kernel_param<> {
    using Float = Float_;
    static constexpr int nColor = nColor_;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    static constexpr QudaReconstructType recon = recon_;
    static constexpr int apeDim = apeDim_;
    typedef typename gauge_mapper<Float,recon>::type Gauge;

    Gauge out;
    const Gauge in;
    Gauge tmp[4];

    int X[4];    // grid dimensions
    int border[4];
    const Float alpha1, alpha2, alpha3;
    const Float tolerance;

    GaugeHYPArg(GaugeField &out, const GaugeField &in, GaugeField *tmp[4], double alpha1, double alpha2, double alpha3) :
      kernel_param(dim3(in.LocalVolumeCB(), 2, apeDim)),
      out(out),
      in(in),
      tmp{*tmp[0], *tmp[1], *tmp[2], *tmp[3]},
      alpha1(alpha1),
      alpha2(alpha2),
      alpha3(alpha3),
      tolerance(in.Precision() == QUDA_DOUBLE_PRECISION ? DOUBLE_TOL : SINGLE_TOL)
    {
      for (int dir = 0; dir < 4; ++dir) {
        border[dir] = in.R()[dir];
        X[dir] = in.X()[dir] - border[dir] * 2;
      }
    }
  };

  template <typename Arg, typename Staple, typename Int>
  __host__ __device__ inline void computeStapleLevel1(const Arg &arg, const int *x, const Int *X, const int parity, const int mu, Staple staple[3], const int dir_ignore)
  {
    using Link = typename get_type<Staple>::type;
    for (int i = 0; i < 3; ++i) staple[i] = Link();

    thread_array<int, 4> dx = { };
    int cnt = -1;
#pragma unroll
    for (int nu = 0; nu < 4; ++nu) {
      // Identify directions orthogonal to the link and
      // ignore the dir_ignore direction (usually the temporal dim
      // when used with STOUT or APE for measurement smearing)
      if (nu == mu) continue;

      cnt += 1;

      {
        // Get link U_{\nu}(x)
        Link U1 = arg.in(nu, linkIndexShift(x, dx, X), parity);

        // Get link U_{\mu}(x+\nu)
        dx[nu]++;
        Link U2 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);
        dx[nu]--;

        // Get link U_{\nu}(x+\mu)
        dx[mu]++;
        Link U3 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);
        dx[mu]--;

        // staple += U_{\nu}(x) * U_{\mu}(x+\nu) * U^\dag_{\nu}(x+\mu)
        staple[cnt] = staple[cnt] + U1 * U2 * conj(U3);
      }

      {
        // Get link U_{\nu}(x-\nu)
        dx[nu]--;
        Link U1 = arg.in(nu, linkIndexShift(x, dx, X), 1 - parity);
        // Get link U_{\mu}(x-\nu)
        Link U2 = arg.in(mu, linkIndexShift(x, dx, X), 1 - parity);

        // Get link U_{\nu}(x-\nu+\mu)
        dx[mu]++;
        Link U3 = arg.in(nu, linkIndexShift(x, dx, X), parity);

        // reset dx
        dx[mu]--;
        dx[nu]++;

        // staple += U^\dag_{\nu}(x-\nu) * U_{\mu}(x-\nu) * U_{\nu}(x-\nu+\mu)
        staple[cnt] = staple[cnt] + conj(U1) * U2 * U3;
      }
    }
  }

  template <typename Arg> struct HYPLevel1 {
    const Arg &arg;
    constexpr HYPLevel1(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;

      // compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      int dx[4] = {0, 0, 0, 0};
      Link U, Stap[3], TestU, I;

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);
      setIdentity(&I);

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStapleLevel1(arg, x, X, parity, dir, Stap, Arg::apeDim);

      for (int i = 0; i < 3; ++i) {
        Stap[i] = Stap[i] * (arg.alpha3 / ((real)2.));
        TestU = I * (static_cast<real>(1.0) - arg.alpha3) + Stap[i] * conj(U);
        polarSu3<real>(TestU, arg.tolerance);
        Stap[i] = TestU * U;
      }

      for (int i = 0; i < 3; ++i) arg.tmp[dir / 2]((dir % 2) * 3 + i, linkIndexShift(x, dx, X), parity) = Stap[i];
    }
  };

  template <typename Arg, typename Staple, typename Int>
  __host__ __device__ inline void computeStapleLevel2(const Arg &arg, const int *x, const Int *X, const int parity, const int mu, Staple staple[3], const int dir_ignore)
  {
    using Link = typename get_type<Staple>::type;
    for (int i = 0; i < 3; ++i) staple[i] = Link();

    thread_array<int, 4> dx = { };
    int cnt = -1;
#pragma unroll
    for (int nu = 0; nu < 4; nu++) {
      // Identify directions orthogonal to the link and
      // ignore the dir_ignore direction (usually the temporal dim
      // when used with STOUT or APE for measurement smearing)
      if (nu == mu) continue;

      cnt += 1;

      for (int rho = 0; rho < 4; ++rho) {
        if (rho == mu || rho == nu) continue;
        int sigma = 0;
        while (sigma == mu || sigma == nu || sigma == rho) sigma += 1;

        {
          // Get link U_{\rho}(x)
          Link U1 = arg.tmp[rho / 2]((rho % 2) * 3 + sigma - (sigma > rho), linkIndexShift(x, dx, X), parity);

          // Get link U_{\mu}(x+\rho)
          dx[rho]++;
          Link U2 = arg.tmp[mu / 2]((mu % 2) * 3 + sigma - (sigma > mu), linkIndexShift(x, dx, X), 1 - parity);
          dx[rho]--;

          // Get link U_{\rho}(x+\mu)
          dx[mu]++;
          Link U3 = arg.tmp[rho / 2]((rho % 2) * 3 + sigma - (sigma > rho), linkIndexShift(x, dx, X), 1 - parity);
          dx[mu]--;

          // staple += U_{\rho}(x) * U_{\mu}(x+\rho) * U^\dag_{\rho}(x+\mu)
          staple[cnt] = staple[cnt] + U1 * U2 * conj(U3);
        }

        {
          // Get link U_{\rho}(x-\rho)
          dx[rho]--;
          Link U1 = arg.tmp[rho / 2]((rho % 2) * 3 + sigma - (sigma > rho), linkIndexShift(x, dx, X), 1 - parity);
          // Get link U_{\mu}(x-\rho)
          Link U2 = arg.tmp[mu / 2]((mu % 2) * 3 + sigma - (sigma > mu), linkIndexShift(x, dx, X), 1 - parity);

          // Get link U_{\rho}(x-\rho+\mu)
          dx[mu]++;
          Link U3 = arg.tmp[rho / 2]((rho % 2) * 3 + sigma - (sigma > rho), linkIndexShift(x, dx, X), parity);

          // reset dx
          dx[mu]--;
          dx[rho]++;

          // staple += U^\dag_{\rho}(x-\rho) * U_{\mu}(x-\rho) * U_{\rho}(x-\rho+\mu)
          staple[cnt] = staple[cnt] + conj(U1) * U2 * U3;
        }
      }
    }
  }

  template <typename Arg> struct HYPLevel2 {
    const Arg &arg;
    constexpr HYPLevel2(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;

      // compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      int dx[4] = {0, 0, 0, 0};
      Link U, Stap[3], TestU, I;

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);
      setIdentity(&I);

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStapleLevel2(arg, x, X, parity, dir, Stap, Arg::apeDim);

      for (int i = 0; i < 3; ++i) {
        Stap[i] = Stap[i] * (arg.alpha2 / ((real)4.));
        TestU = I * (static_cast<real>(1.0) - arg.alpha2) + Stap[i] * conj(U);
        polarSu3<real>(TestU, arg.tolerance);
        Stap[i] = TestU * U;
      }

      for (int i = 0; i < 3; ++i) arg.tmp[dir / 2 + 2]((dir % 2) * 3 + i, linkIndexShift(x, dx, X), parity) = Stap[i];
    }
  };

  template <typename Arg, typename Staple, typename Int>
  __host__ __device__ inline void computeStapleLevel3(const Arg &arg, const int *x, const Int *X, const int parity, const int mu, Staple &staple, const int dir_ignore)
  {
    using Link = typename get_type<Staple>::type;
    staple = Link();

    thread_array<int, 4> dx = { };
#pragma unroll
    for (int nu = 0; nu < 4 ; nu++) {
      // Identify directions orthogonal to the link and
      // ignore the dir_ignore direction (usually the temporal dim
      // when used with STOUT or APE for measurement smearing)
      if (nu == mu) continue;

      {
        // Get link U_{\nu}(x)
        Link U1 = arg.tmp[nu / 2 + 2]((nu % 2) * 3 + mu - (mu > nu), linkIndexShift(x, dx, X), parity);

        // Get link U_{\mu}(x+\nu)
        dx[nu]++;
        Link U2 = arg.tmp[mu / 2 + 2]((mu % 2) * 3 + nu - (nu > mu), linkIndexShift(x, dx, X), 1 - parity);
        dx[nu]--;

        // Get link U_{\nu}(x+\mu)
        dx[mu]++;
        Link U3 = arg.tmp[nu / 2 + 2]((nu % 2) * 3 + mu - (mu > nu), linkIndexShift(x, dx, X), 1 - parity);
        dx[mu]--;

        // staple += U_{\nu}(x) * U_{\mu}(x+\nu) * U^\dag_{\nu}(x+\mu)
        staple = staple + U1 * U2 * conj(U3);
      }

      {
        // Get link U_{\nu}(x-\nu)
        dx[nu]--;
        Link U1 = arg.tmp[nu / 2 + 2]((nu % 2) * 3 + mu - (mu > nu), linkIndexShift(x, dx, X), 1 - parity);
        // Get link U_{\mu}(x-\nu)
        Link U2 = arg.tmp[mu / 2 + 2]((mu % 2) * 3 + nu - (nu > mu), linkIndexShift(x, dx, X), 1 - parity);

        // Get link U_{\nu}(x-\nu+\mu)
        dx[mu]++;
        Link U3 = arg.tmp[nu / 2 + 2]((nu % 2) * 3 + mu - (mu > nu), linkIndexShift(x, dx, X), parity);

        // reset dx
        dx[mu]--;
        dx[nu]++;

        // staple += U^\dag_{\nu}(x-\nu) * U_{\mu}(x-\nu) * U_{\nu}(x-\nu+\mu)
        staple = staple + conj(U1) * U2 * U3;
      }
    }
  }

  template <typename Arg> struct HYPLevel3 {
    const Arg &arg;
    constexpr HYPLevel3(const Arg &arg) : arg(arg) {}
    static constexpr const char* filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::Float;
      typedef Matrix<complex<real>, Arg::nColor> Link;

      // compute spacetime and local coords
      int X[4];
      for (int dr = 0; dr < 4; ++dr) X[dr] = arg.X[dr];
      int x[4];
      getCoords(x, x_cb, X, parity);
      for (int dr = 0; dr < 4; ++dr) {
        x[dr] += arg.border[dr];
        X[dr] += 2 * arg.border[dr];
      }

      int dx[4] = {0, 0, 0, 0};
      Link U, Stap, TestU, I;

      // Get link U
      U = arg.in(dir, linkIndexShift(x, dx, X), parity);
      setIdentity(&I);

      // This function gets stap = S_{mu,nu} i.e., the staple of length 3,
      computeStapleLevel3(arg, x, X, parity, dir, Stap, Arg::apeDim);

      Stap = Stap * (arg.alpha1 / ((real)6.));
      TestU = I * (static_cast<real>(1.0) - arg.alpha1) + Stap * conj(U);
      polarSu3<real>(TestU, arg.tolerance);
      Stap = TestU * U;

      arg.out(dir, linkIndexShift(x, dx, X), parity) = Stap;
    }
  };
} // namespace quda
