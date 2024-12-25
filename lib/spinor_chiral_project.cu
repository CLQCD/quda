#include <color_spinor_field.h>
#include <tunable_nd.h>
#include <kernels/spinor_chiral_project.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Float, int Nc, QudaChirality Chirality> class SpinorChiralEmbed : TunableKernel2D
  {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    template <template <int, int, QudaChirality, bool> class Basis>
    using Arg = ChrialEmbedSpinorArg<Float, Nc, Chirality, Basis>;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    SpinorChiralEmbed(ColorSpinorField &out, const ColorSpinorField &in) :
      TunableKernel2D(in, in.SiteSubset()), in(in), out(out)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS) {
        launch<ChiralEmbedSpinor>(tp, stream, Arg<ChiralToNonRelBasis>(out, in));
      } else if (out.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS && in.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS) {
        launch<ChiralEmbedSpinor>(tp, stream, Arg<NonRelBasis>(out, in));
      } else {
        errorQuda("Unsupported combination of dst gamma basis %d and src gamma basis %d", out.GammaBasis(),
                  in.GammaBasis());
      }
    }

    long long bytes() const { return in.Bytes() + out.Bytes(); }
  };

  void spinorChiralEmbed(ColorSpinorField &dst, const ColorSpinorField &src, QudaChirality chirality)
  {
    checkPrecision(dst, src);

    if (src.Nspin() != 2) { errorQuda("Unsupported src nSpin=%d", src.Nspin()); }
    if (dst.Nspin() != 4) { errorQuda("Unsupported dst nSpin=%d", dst.Nspin()); }

    if (dst.Ncolor() == 3 && src.Ncolor() == 3) {
      if (src.Precision() == QUDA_DOUBLE_PRECISION) {
        if (chirality == QUDA_CHIRALITY_UPPER) {
          SpinorChiralEmbed<double, 3, QUDA_CHIRALITY_UPPER>(dst, src);
        } else if (chirality == QUDA_CHIRALITY_LOWER) {
          SpinorChiralEmbed<double, 3, QUDA_CHIRALITY_LOWER>(dst, src);
        } else {
          errorQuda("Unsupported chirality %d", chirality);
        }
      } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
        if (chirality == QUDA_CHIRALITY_UPPER) {
          SpinorChiralEmbed<float, 3, QUDA_CHIRALITY_UPPER>(dst, src);
        } else if (chirality == QUDA_CHIRALITY_LOWER) {
          SpinorChiralEmbed<float, 3, QUDA_CHIRALITY_LOWER>(dst, src);
        } else {
          errorQuda("Unsupported chirality %d", chirality);
        }
      } else {
        errorQuda("Precision %d not implemented", src.Precision());
      }
    } else {
      errorQuda("nColor=%d not implemented", src.Ncolor());
    }
  }

  template <typename Float, int Nc, QudaChirality Chirality> class SpinorChiralProject : TunableKernel2D
  {
    ColorSpinorField &out;
    const ColorSpinorField &in;
    template <template <int, int, QudaChirality, bool> class Basis>
    using Arg = ChiralProjectSpinorArg<Float, Nc, Chirality, Basis>;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    SpinorChiralProject(ColorSpinorField &out, const ColorSpinorField &in) :
      TunableKernel2D(in, in.SiteSubset()), in(in), out(out)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (out.GammaBasis() == QUDA_CHIRAL_GAMMA_BASIS && in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS) {
        launch<ChiralProjectSpinor>(tp, stream, Arg<NonRelToChiralBasis>(out, in));
      } else if (out.GammaBasis() == QUDA_DEGRAND_ROSSI_GAMMA_BASIS && in.GammaBasis() == QUDA_UKQCD_GAMMA_BASIS) {
        launch<ChiralProjectSpinor>(tp, stream, Arg<RelBasis>(out, in));
      } else {
        errorQuda("Unsupported combination of dst gamma basis %d and src gamma basis %d", out.GammaBasis(),
                  in.GammaBasis());
      }
    }

    long long bytes() const { return in.Bytes() + out.Bytes(); }
  };

  void spinorChiralProject(ColorSpinorField &dst, const ColorSpinorField &src, QudaChirality chirality)
  {
    checkPrecision(dst, src);

    if (src.Nspin() != 4) { errorQuda("Unsupported src nSpin=%d", src.Nspin()); }
    if (dst.Nspin() != 2) { errorQuda("Unsupported dst nSpin=%d", dst.Nspin()); }

    if (dst.Ncolor() == 3 && src.Ncolor() == 3) {
      if (src.Precision() == QUDA_DOUBLE_PRECISION) {
        if (chirality == QUDA_CHIRALITY_UPPER) {
          SpinorChiralProject<double, 3, QUDA_CHIRALITY_UPPER>(dst, src);
        } else if (chirality == QUDA_CHIRALITY_LOWER) {
          SpinorChiralProject<double, 3, QUDA_CHIRALITY_LOWER>(dst, src);
        } else {
          errorQuda("Unsupported chirality %d", chirality);
        }
      } else if (src.Precision() == QUDA_SINGLE_PRECISION) {
        if (chirality == QUDA_CHIRALITY_UPPER) {
          SpinorChiralProject<float, 3, QUDA_CHIRALITY_UPPER>(dst, src);
        } else if (chirality == QUDA_CHIRALITY_LOWER) {
          SpinorChiralProject<float, 3, QUDA_CHIRALITY_LOWER>(dst, src);
        } else {
          errorQuda("Unsupported chirality %d", chirality);
        }
      } else {
        errorQuda("Precision %d not implemented", src.Precision());
      }
    } else {
      errorQuda("nColor=%d not implemented", src.Ncolor());
    }
  }

} // namespace quda
