#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_improved_wilson.cuh>

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{

  template <typename Arg> class ImprovedWilson : public Dslash<improvedWilson, Arg>
  {
    using Dslash = Dslash<improvedWilson, Arg>;

  public:
    ImprovedWilson(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in)
    {
      if(in.Ndim() == 5) {
        TunableKernel3D::resizeVector(in.X(4), arg.nParity);
      }
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::template instantiate<packShmem>(tp, stream);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct ImprovedWilsonApply {

    inline ImprovedWilsonApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L,
                               double a, const ColorSpinorField &x, double improve, int parity, bool dagger, const int *comm_override,
                               TimeProfile &profile)
    {
      constexpr int nDim = 4;
      ImprovedWilsonArg<Float, nColor, nDim, recon> arg(out, in, U, L, a, x, improve, parity, dagger, comm_override);
      ImprovedWilson<decltype(arg)> wilson(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, in.VolumeCB(), in.GhostFaceCB(), profile);
    }
  };

  // Apply the Wilson operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the a normalization for the Wilson operator.
#ifdef GPU_WILSON_DIRAC
  void ApplyImprovedWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L,
                           double a, const ColorSpinorField &x, double improve, int parity, bool dagger, const int *comm_override,
                           TimeProfile &profile)
  {
    instantiate<ImprovedWilsonApply, WilsonReconstruct>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
  }
#else
  void ApplyImprovedWilson(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, const GaugeField &,
                           double, const ColorSpinorField &, double, int, bool, const int *,
                           TimeProfile &)
  {
    errorQuda("Improved Wilson dslash has not been built");
  }
#endif // GPU_WILSON_DIRAC

} // namespace quda
