#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_stout_force.cuh>

namespace quda
{
  template <typename Float, int nColor, QudaReconstructType recon> class GaugeSTOUTForceLambda : TunableKernel3D
  {
    GaugeField &lambda;
    const GaugeField &sigma;
    const GaugeField &gauge;
    const Float rho;
    const int dir_ignore;
    const Float anisotropy;
    const int stoutDim;
    unsigned int minThreads() const { return gauge.LocalVolumeCB(); }

  public:
    GaugeSTOUTForceLambda(const GaugeField &gauge, GaugeField &lambda, const GaugeField &sigma, double rho,
                          int dir_ignore, double anisotropy) :
      TunableKernel3D(gauge, 2, (dir_ignore == 4) ? 4 : 3),
      lambda(lambda),
      sigma(sigma),
      gauge(gauge),
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
        launch<STOUTForceLambda>(
          tp, stream, STOUTForceLambdaArg<Float, nColor, recon, 3>(lambda, sigma, gauge, rho, dir_ignore, anisotropy));
      } else if (stoutDim == 4) {
        launch<STOUTForceLambda>(
          tp, stream, STOUTForceLambdaArg<Float, nColor, recon, 4>(lambda, sigma, gauge, rho, dir_ignore, anisotropy));
      }
    }

    void preTune() { lambda.backup(); }
    void postTune() { lambda.restore(); }

    long long flops() const // just counts matrix multiplication
    {
      auto mat_flops = gauge.Ncolor() * gauge.Ncolor() * (8ll * gauge.Ncolor() - 2ll);
      return (2 + (stoutDim - 1) * 4) * mat_flops * stoutDim * gauge.LocalVolume();
    }

    long long bytes() const // 6 links per dim, 1 in, 1 out.
    {
      return ((1 + (stoutDim - 1) * 6) * gauge.Reconstruct() * gauge.Precision()
              + lambda.Reconstruct() * lambda.Precision())
        * stoutDim * gauge.LocalVolume();
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeSTOUTForceSigma : TunableKernel3D
  {
    GaugeField &sigma;
    const GaugeField &lambda;
    const GaugeField &gauge;
    const Float rho;
    const int dir_ignore;
    const Float anisotropy;
    const int stoutDim;
    unsigned int minThreads() const { return gauge.LocalVolumeCB(); }

  public:
    GaugeSTOUTForceSigma(const GaugeField &gauge, GaugeField &sigma, const GaugeField &lambda, double rho,
                         int dir_ignore, double anisotropy) :
      TunableKernel3D(gauge, 2, (dir_ignore == 4) ? 4 : 3),
      sigma(sigma),
      lambda(lambda),
      gauge(gauge),
      rho(static_cast<Float>(rho)),
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
        launch<STOUTForceSigma>(
          tp, stream, STOUTForceSigmaArg<Float, nColor, recon, 3>(sigma, lambda, gauge, rho, dir_ignore, anisotropy));
      } else if (stoutDim == 4) {
        launch<STOUTForceSigma>(
          tp, stream, STOUTForceSigmaArg<Float, nColor, recon, 4>(sigma, lambda, gauge, rho, dir_ignore, anisotropy));
      }
    }

    void preTune() { sigma.backup(); }
    void postTune() { sigma.restore(); }

    long long flops() const // just counts matrix multiplication
    {
      auto mat_flops = gauge.Ncolor() * gauge.Ncolor() * (8ll * gauge.Ncolor() - 2ll);
      return (2 + (stoutDim - 1) * (4)) * mat_flops * stoutDim * gauge.LocalVolume();
    }

    long long bytes() const // 6 links per dim, 1 in, 1 out.
    {
      return ((1 + (stoutDim - 1) * 6) * gauge.Reconstruct() * gauge.Precision() + sigma.Reconstruct() * sigma.Precision())
        * stoutDim * gauge.LocalVolume();
    }
  };

  void STOUTForceStep(GaugeField &force, GaugeField &lambda, GaugeField &gauge, double rho, int dir_ignore,
                      double smear_anisotropy)
  {
    checkPrecision(force, lambda, gauge);
    // checkReconstruct(force, lambda, gauge);
    checkNative(force, lambda, gauge);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeSTOUTForceLambda>(gauge, lambda, force, rho, dir_ignore, smear_anisotropy);
    lambda.exchangeExtendedGhost(lambda.R());
    instantiate<GaugeSTOUTForceSigma>(gauge, force, lambda, rho, dir_ignore, smear_anisotropy);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
