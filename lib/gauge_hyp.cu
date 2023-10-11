#include <quda_internal.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_hyp.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeHYPLevel1 : TunableKernel3D
  {
    static constexpr int apeDim = 3; // apply APE in space only
    GaugeField &out;
    const GaugeField &in;
    GaugeField *tmp[4];
    const Float alpha1, alpha2, alpha3;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }

  public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeHYPLevel1(GaugeField &out, const GaugeField &in, GaugeField *tmp[4], double alpha1, double alpha2, double alpha3) :
      TunableKernel3D(in, 2, apeDim),
      out(out),
      in(in),
      tmp{tmp[0], tmp[1], tmp[2], tmp[3]},
      alpha1(static_cast<Float>(alpha1)),
      alpha2(static_cast<Float>(alpha2)),
      alpha3(static_cast<Float>(alpha3))
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<HYPLevel1>(tp, stream, GaugeHYPArg<Float,nColor,recon, apeDim>(out, in, tmp, alpha1, alpha2, alpha3));
    }

    void preTune() { out.backup(); } // defensive measure in case they alias
    void postTune() { out.restore(); }

    long long flops() const
    {
      auto mat_flops = in.Ncolor() * in.Ncolor() * (8ll * in.Ncolor() - 2ll);
      return (2 + (apeDim - 1) * 4) * mat_flops * apeDim * in.LocalVolume();
    }

    long long bytes() const // 6 links per dim, 1 in, 1 out.
    {
      return ((1 + (apeDim - 1) * 6) * in.Reconstruct() * in.Precision() +
              out.Reconstruct() * out.Precision()) * apeDim * in.LocalVolume();
    }

  }; // GaugeAPE

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeHYPLevel2 : TunableKernel3D
  {
    static constexpr int apeDim = 3; // apply APE in space only
    GaugeField &out;
    const GaugeField &in;
    GaugeField *tmp[4];
    const Float alpha1, alpha2, alpha3;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }

  public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeHYPLevel2(GaugeField &out, const GaugeField &in, GaugeField *tmp[4], double alpha1, double alpha2, double alpha3) :
      TunableKernel3D(in, 2, apeDim),
      out(out),
      in(in),
      tmp{tmp[0], tmp[1], tmp[2], tmp[3]},
      alpha1(static_cast<Float>(alpha1)),
      alpha2(static_cast<Float>(alpha2)),
      alpha3(static_cast<Float>(alpha3))
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<HYPLevel2>(tp, stream, GaugeHYPArg<Float,nColor,recon, apeDim>(out, in, tmp, alpha1, alpha2, alpha3));
    }

    void preTune() { out.backup(); } // defensive measure in case they alias
    void postTune() { out.restore(); }

    long long flops() const
    {
      auto mat_flops = in.Ncolor() * in.Ncolor() * (8ll * in.Ncolor() - 2ll);
      return (2 + (apeDim - 1) * 4) * mat_flops * apeDim * in.LocalVolume();
    }

    long long bytes() const // 6 links per dim, 1 in, 1 out.
    {
      return ((1 + (apeDim - 1) * 6) * in.Reconstruct() * in.Precision() +
              out.Reconstruct() * out.Precision()) * apeDim * in.LocalVolume();
    }

  }; // GaugeAPE

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeHYPLevel3 : TunableKernel3D
  {
    static constexpr int apeDim = 3; // apply APE in space only
    GaugeField &out;
    const GaugeField &in;
    GaugeField *tmp[4];
    const Float alpha1, alpha2, alpha3;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }

  public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeHYPLevel3(GaugeField &out, const GaugeField &in, GaugeField *tmp[4], double alpha1, double alpha2, double alpha3) :
      TunableKernel3D(in, 2, apeDim),
      out(out),
      in(in),
      tmp{tmp[0], tmp[1], tmp[2], tmp[3]},
      alpha1(static_cast<Float>(alpha1)),
      alpha2(static_cast<Float>(alpha2)),
      alpha3(static_cast<Float>(alpha3))
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<HYPLevel3>(tp, stream, GaugeHYPArg<Float,nColor,recon, apeDim>(out, in, tmp, alpha1, alpha2, alpha3));
    }

    void preTune() { out.backup(); } // defensive measure in case they alias
    void postTune() { out.restore(); }

    long long flops() const
    {
      auto mat_flops = in.Ncolor() * in.Ncolor() * (8ll * in.Ncolor() - 2ll);
      return (2 + (apeDim - 1) * 4) * mat_flops * apeDim * in.LocalVolume();
    }

    long long bytes() const // 6 links per dim, 1 in, 1 out.
    {
      return ((1 + (apeDim - 1) * 6) * in.Reconstruct() * in.Precision() +
              out.Reconstruct() * out.Precision()) * apeDim * in.LocalVolume();
    }

  }; // GaugeAPE

  void HYPStep(GaugeField &out, GaugeField& in, GaugeField *tmp[4], double alpha1, double alpha2, double alpha3)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    instantiate<GaugeHYPLevel1>(out, in, tmp, alpha1, alpha2, alpha3);
    tmp[0]->exchangeExtendedGhost(tmp[0]->R(), false);
    tmp[1]->exchangeExtendedGhost(tmp[1]->R(), false);
    instantiate<GaugeHYPLevel2>(out, in, tmp, alpha1, alpha2, alpha3);
    tmp[2]->exchangeExtendedGhost(tmp[2]->R(), false);
    tmp[3]->exchangeExtendedGhost(tmp[3]->R(), false);
    instantiate<GaugeHYPLevel3>(out, in, tmp, alpha1, alpha2, alpha3);
    out.exchangeExtendedGhost(out.R(), false);
  }

}
