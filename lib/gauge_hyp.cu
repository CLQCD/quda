#include <quda_internal.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_hyp.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeHYP : TunableKernel3D
  {
    static constexpr int hypDim = 4; // apply HYP in 4 dimensions
    GaugeField &out;
    const GaugeField &in;
    GaugeField *tmp[4];
    const Float alpha;
    const int level;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }

  public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeHYP(GaugeField &out, const GaugeField &in, GaugeField *tmp[4], double alpha, int level) :
      TunableKernel3D(in, 2, hypDim),
      out(out),
      in(in),
      tmp{tmp[0], tmp[1], tmp[2], tmp[3]},
      alpha(static_cast<Float>(alpha)),
      level(level)
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (level == 1) {
        launch<HYP>(tp, stream, GaugeHYPArg<Float,nColor,recon,hypDim,1>(out, in, tmp, alpha));
      } else if (level == 2) {
        launch<HYP>(tp, stream, GaugeHYPArg<Float,nColor,recon,hypDim,2>(out, in, tmp, alpha));
      } else if (level == 3) {
        launch<HYP>(tp, stream, GaugeHYPArg<Float,nColor,recon,hypDim,3>(out, in, tmp, alpha));
      }
    }

    void preTune() { out.backup(); } // defensive measure in case they alias
    void postTune() { out.restore(); }

    long long flops() const
    {
      auto mat_flops = in.Ncolor() * in.Ncolor() * (8ll * in.Ncolor() - 2ll);
      return (2 + (hypDim - 1) * 4) * mat_flops * hypDim * in.LocalVolume();
    }

    long long bytes() const // 6 links per dim, 1 in, 1 out.
    {
      return ((1 + (hypDim - 1) * 6) * in.Reconstruct() * in.Precision() +
              out.Reconstruct() * out.Precision()) * hypDim * in.LocalVolume();
    }

  }; // GaugeAPE

  void HYPStep(GaugeField &out, GaugeField& in, GaugeField *tmp[4], double alpha1, double alpha2, double alpha3)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    instantiate<GaugeHYP>(out, in, tmp, alpha3, 1);
    tmp[0]->exchangeExtendedGhost(tmp[0]->R(), false);
    tmp[1]->exchangeExtendedGhost(tmp[1]->R(), false);
    instantiate<GaugeHYP>(out, in, tmp, alpha2, 2);
    tmp[2]->exchangeExtendedGhost(tmp[2]->R(), false);
    tmp[3]->exchangeExtendedGhost(tmp[3]->R(), false);
    instantiate<GaugeHYP>(out, in, tmp, alpha1, 3);
    out.exchangeExtendedGhost(out.R(), false);
  }

}
