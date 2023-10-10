#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_improved_wilson_clover.cuh>

/**
   This is the Wilson-clover linear operator
*/

namespace quda
{

  template <typename Arg> class ImprovedWilsonClover : public Dslash<improvedWilsonClover, Arg>
  {
    using Dslash = Dslash<improvedWilsonClover, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    ImprovedWilsonClover(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) {}

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      if (arg.xpay)
        Dslash::template instantiate<packShmem, true>(tp, stream);
      else
        errorQuda("Improved Wilson-clover operator only defined for xpay=true");
    }

    long long flops() const
    {
      int clover_flops = 504;
      long long flops = Dslash::flops();

      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: flops += clover_flops * in.Volume(); break;
      default: break; // all clover flops are in the interior kernel
      }
      return flops;
    }

    long long bytes() const
    {
      int clover_bytes = 72 * in.Precision() + (isFixed<typename Arg::Float>::value ? 2 * sizeof(float) : 0);
      long long bytes = Dslash::bytes();

      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: bytes += clover_bytes * in.Volume(); break;
      default: break;
      }

      return bytes;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct ImprovedWilsonCloverApply {

    inline ImprovedWilsonCloverApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L, const CloverField &A,
        double a, const ColorSpinorField &x, double improve, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      ImprovedWilsonCloverArg<Float, nColor, nDim, recon> arg(out, in, U, L, A, a, 0.0, x, improve, parity, dagger, comm_override);
      ImprovedWilsonClover<decltype(arg)> wilson(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, in.VolumeCB(), in.GhostFaceCB(), profile);
    }
  };

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x)*in(x) + a * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
#ifdef GPU_CLOVER_DIRAC
  void ApplyImprovedWilsonClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                 const GaugeField &L, const CloverField &A, double a,
                                 const ColorSpinorField &x, double improve, int parity, bool dagger, const int *comm_override,
                                 TimeProfile &profile)
  {
    instantiate<ImprovedWilsonCloverApply>(out, in, U, L, A, a, x, improve, parity, dagger, comm_override, profile);
  }
#else
  void ApplyImprovedWilsonClover(ColorSpinorField &, const ColorSpinorField &, const GaugeField &,
                                 const GaugeField &, const CloverField &, double,
                                 const ColorSpinorField &, double, int, bool, const int *,
                                 TimeProfile &)
  {
    errorQuda("Improved clover dslash has not been built");
  }
#endif

} // namespace quda
