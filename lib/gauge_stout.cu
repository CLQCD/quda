#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_stout.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeSTOUT : TunableKernel3D
  {
    GaugeField &out;
    const GaugeField &in;
    const Float rho;
    const int dir_ignore;
    const Float anisotropy;
    const int stoutDim;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }

  public:
    // (2,3/4): 2 for parity in the y thread dim, 3 or 4 corresponds to mapping direction to the z thread dim
    GaugeSTOUT(GaugeField &out, const GaugeField &in, double rho, int dir_ignore, double anisotropy) :
      TunableKernel3D(in, 2, (dir_ignore == 4) ? 4 : 3),
      out(out),
      in(in),
      rho(rho),
      dir_ignore(dir_ignore),
      anisotropy(anisotropy),
      stoutDim((dir_ignore == 4) ? 4 : 3)
    {
      strcat(aux, ",dir_ignore=");
      i32toa(aux + strlen(aux), dir_ignore);
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (stoutDim == 3) {
        launch<STOUT>(tp, stream, STOUTArg<Float, nColor, recon, 3>(out, in, rho, dir_ignore, anisotropy));
      } else if (stoutDim == 4) {
        launch<STOUT>(tp, stream, STOUTArg<Float, nColor, recon, 4>(out, in, rho, dir_ignore, anisotropy));
      }
    }

    void preTune()
    {
      if (out.data() == in.data()) out.backup();
    }
    void postTune()
    {
      if (out.data() == in.data()) out.restore();
    }

    long long flops() const // just counts matrix multiplication
    {
      auto mat_flops = in.Ncolor() * in.Ncolor() * (8ll * in.Ncolor() - 2ll);
      return (2 + (stoutDim - 1) * 4) * mat_flops * stoutDim * in.LocalVolume();
    }

    long long bytes() const // 6 links per dim, 1 in, 1 out.
    {
      const long long in_lp
        = static_cast<long long>(static_cast<int>(in.Reconstruct()) * static_cast<int>(in.Precision()));
      const long long out_lp
        = static_cast<long long>(static_cast<int>(out.Reconstruct()) * static_cast<int>(out.Precision()));
      return ((1 + (stoutDim - 1) * (improved ? 24 : 6)) * in_lp + out_lp) * stoutDim * in.LocalVolume();
    }
  };

  void STOUTStep(GaugeField &out, GaugeField &in, double rho, int dir_ignore, double smear_anisotropy)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeSTOUT>(out, in, rho, dir_ignore, smear_anisotropy);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    out.exchangeExtendedGhost(out.R(), false);
  }

} // namespace quda
