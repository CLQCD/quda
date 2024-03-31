#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_overlap.cuh>

/**
   This is the Wilson-clover linear operator
*/

namespace quda
{

  template <typename Arg> class Overlap : public Dslash<overlap, Arg>
  {
    using Dslash = Dslash<overlap, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    Overlap(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) { }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::template instantiate<packShmem>(tp, stream);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct OverlapApply {

    inline OverlapApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                        const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                        TimeProfile &profile)
    {
      constexpr int nDim = 4;
      OverlapArg<Float, nColor, nDim, recon> arg(out, in, U, a, x, parity, dagger, comm_override);
      Overlap<decltype(arg)> overlap(arg, out, in);

      dslash::DslashPolicyTune<decltype(overlap)> policy(overlap, in, in.VolumeCB(), in.GhostFaceCB(), profile);
    }
  };

  // Apply the overlap operator
  // out(x) = M*in = (A(x)*in(x) + a * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
#ifdef GPU_WILSON_DIRAC
  void ApplyOverlap(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                    const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    instantiate<OverlapApply>(out, in, U, a, x, parity, dagger, comm_override, profile);
  }
#else
  void ApplyOverlap(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, double, const ColorSpinorField &,
                    int, bool, const int *, TimeProfile &)
  {
    errorQuda("Wilson dslash has not been built");
  }
#endif

} // namespace quda
