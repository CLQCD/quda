#include <math_helper.cuh>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <kernel.h>
#include "spinor_chiral_project_helper.cuh"

namespace quda
{
  using namespace colorspinor;

  template <typename store_t, int nColor_, QudaChirality Chirality_, template <int, int, QudaChirality, bool> class Basis_>
  struct ChiralProjectSpinorArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nSpinOut = 2;
    static constexpr int nSpinIn = 4;
    static constexpr int nColor = nColor_;
    static constexpr QudaChirality Chirality = Chirality_;
    using Basis = Basis_<4, nColor_, Chirality_, true>;
    using Vout = typename colorspinor_mapper<store_t, nSpinOut, nColor>::type;
    using Vin = typename colorspinor_mapper<store_t, nSpinIn, nColor>::type;

    int X[4];
    Vout out;
    const Vin in;
    ChiralProjectSpinorArg(ColorSpinorField &out, const ColorSpinorField &in) :
      kernel_param(dim3(in.VolumeCB(), in.SiteSubset(), 1)), out(out), in(in)
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = in.X()[dir];
      X[0] *= (in.SiteSubset() == 1) ? 2 : 1; // need full lattice dims
    }
  };

  template <typename store_t, int nColor_, QudaChirality Chirality_, template <int, int, QudaChirality, bool> class Basis_>
  struct ChrialEmbedSpinorArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nSpinOut = 4;
    static constexpr int nSpinIn = 2;
    static constexpr int nColor = nColor_;
    static constexpr QudaChirality Chirality = Chirality_;
    using Basis = Basis_<4, nColor_, Chirality_, false>;
    using Vout = typename colorspinor_mapper<store_t, nSpinOut, nColor>::type;
    using Vin = typename colorspinor_mapper<store_t, nSpinIn, nColor>::type;

    int X[4];
    Vout out;
    const Vin in;
    ChrialEmbedSpinorArg(ColorSpinorField &out, const ColorSpinorField &in) :
      kernel_param(dim3(in.VolumeCB(), in.SiteSubset(), 1)), out(out), in(in)
    {
      for (int dir = 0; dir < 4; dir++) X[dir] = in.X()[dir];
      X[0] *= (in.SiteSubset() == 1) ? 2 : 1; // need full lattice dims
    }
  };

  template <typename Arg> struct ChiralEmbedSpinor {
    const Arg &arg;
    constexpr ChiralEmbedSpinor(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      using VectorOut = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpinOut>;
      using VectorIn = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpinIn>;
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      VectorOut out;
      VectorIn in = arg.in(x_cb, parity);
      typename Arg::Basis basis;
      basis(out.data, in.data);
      arg.out(x_cb, parity) = out;
    }
  };

  template <typename Arg> struct ChiralProjectSpinor {
    const Arg &arg;
    constexpr ChiralProjectSpinor(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int parity)
    {
      using VectorOut = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpinOut>;
      using VectorIn = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpinIn>;
      int x[4];
      getCoords(x, x_cb, arg.X, parity);
      VectorOut out;
      VectorIn in = arg.in(x_cb, parity);
      typename Arg::Basis basis;
      basis(out.data, in.data);
      arg.out(x_cb, parity) = out;
    }
  };

} // namespace quda
