#pragma once

#include <memory>
#include <dirac_quda.h>
#include <staggered_rotating_links.h>

namespace quda
{
  class StaggeredRotatingHalo;

  /**
     Build the optional path cache from the extended level-2 X links.
     @param[out] cache Four named destination fields for VXXTau and VXYT paths.
     @param[in] fat Extended level-2 X links, including x/y/t torus halos.
   */
  void BuildHISQRotatingOrbitalSpinLinkCache(const MutableHISQRotatingOrbitalSpinLinkCache &cache, const GaugeField &fat);

  /**
     Add the rotating correction to an already-computed improved-staggered
     Dslash. The caller owns the rotating gauge fields and reusable
     corner-halo workspace. A one-field gauge list supplies extended level-2
     X links for cache-off operation; a four-field list supplies the named
     VXXTau(-t), VXXTau(+t), VXYT(-t), and VXYT(+t) cache fields. Full and
     parity spinors are both supported.

     The shift-center-half coordinate midpoint is derived as
     (global Lx / 2, global Ly / 2). Production callers and all validation use
     this geometric midpoint on even lattices. For an MPI-partitioned x, y,
     or t direction, comm_override must not disable communication in that
     direction, since the rotating stencil accesses off-rank corner
     endpoints. The stencil has no z spinor displacement.

     @param[in,out] out Output spinors already containing the ordinary HISQ Dslash.
     @param[in] in Input spinors.
     @param[in] fields One extended level-2 X field or four named path-cache fields.
     @param[in] parity Output parity, or QUDA_INVALID_PARITY for a full field.
     @param[in] dagger Whether to apply the daggered rotating correction.
     @param[in] angular_velocity Rotation angular velocity in lattice units.
     @param[in] comm_override Optional per-dimension communication mask. An
     MPI-partitioned x, y, or t direction must remain enabled.
     @param[in,out] spinor_halo Optional reusable diagonal spinor-halo workspace.
   */
  void ApplyStaggeredRotating(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                               const StaggeredRotatingGaugeFieldList &fields, int parity, bool dagger,
                               double angular_velocity, const int *comm_override,
                               StaggeredRotatingHalo *spinor_halo = nullptr);

  // The rotation tag selects these subclasses in a factory branch preceding
  // the ordinary ASQTAD dispatch.  They retain the parent-reported type:
  // the currently supported multigrid hierarchy has an ordinary HISQ coarse
  // operator, while its fine smoother remains rotation aware.

  /**
     Full rotating HISQ operator.

     The ordinary parent applies the post-epsilon fat and Naik links.  This
     subclass then adds the rotation term built from the separately resident
     level-2 X links, matching the ordering used by CLGLib.
   */
  class DiracImprovedStaggeredRotating : public DiracImprovedStaggered
  {
  protected:
    double angular_velocity;                                   /** Rotation angular velocity in lattice units. */
    StaggeredRotatingGaugeFieldList rotating_fields;           /** Non-owning native X-link or path-cache view. */
    mutable std::unique_ptr<StaggeredRotatingHalo> spinor_halo; /** Owned reusable off-rank spinor workspace. */

  public:
    /** @param[in] param Improved-staggered fields and rotating dispatch parameters. */
    DiracImprovedStaggeredRotating(const DiracParam &param);
    /** @param[in] dirac Operator to copy; the reusable halo is not shared. */
    DiracImprovedStaggeredRotating(const DiracImprovedStaggeredRotating &dirac);
    ~DiracImprovedStaggeredRotating() override;
    /** @param[in] dirac Operator state to copy; @return This operator. */
    DiracImprovedStaggeredRotating &operator=(const DiracImprovedStaggeredRotating &dirac);

    /**
       Refresh ordinary and rotating resident-field pointers after a gauge reload.
       @param[in] gauge Thin gauge, unused by improved staggered operators.
       @param[in] fat_gauge_in Post-epsilon fat links at this operator's precision.
       @param[in] long_gauge_in Naik links at this operator's precision.
       @param[in] clover Clover field, unused by improved staggered operators.
     */
    void updateFields(GaugeField *gauge, GaugeField *fat_gauge_in, GaugeField *long_gauge_in,
                      CloverField *clover) override;

    /** @param[out] out Dslash result. @param[in] in Input spinors. @param[in] parity Output parity. */
    void Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                QudaParity parity) const override;
    /**
       Apply Dslash followed by the parent's xpay convention.
       @param[out] out Result.
       @param[in] in Dslash input.
       @param[in] parity Output parity.
       @param[in] x Xpay vector.
       @param[in] k Xpay scale.
     */
    void DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, QudaParity parity,
                    cvector_ref<const ColorSpinorField> &x, double k) const override;
    /** @param[out] out Matrix result. @param[in] in Full-field input. */
    void M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const override;
  };

  /**
     Even-odd preconditioned rotating HISQ operator.

     Keeping this as a subclass lets QUDA's existing Schur-complement,
     multi-shift, and multigrid solver machinery consume the rotating Dslash.
   */
  class DiracImprovedStaggeredRotatingPC : public DiracImprovedStaggeredPC
  {
  protected:
    double angular_velocity;                                   /** Rotation angular velocity in lattice units. */
    StaggeredRotatingGaugeFieldList rotating_fields;           /** Non-owning native X-link or path-cache view. */
    mutable std::unique_ptr<StaggeredRotatingHalo> spinor_halo; /** Owned reusable off-rank spinor workspace. */

  public:
    /** @param[in] param Preconditioned improved-staggered and rotating parameters. */
    DiracImprovedStaggeredRotatingPC(const DiracParam &param);
    /** @param[in] dirac Operator to copy; the reusable halo is not shared. */
    DiracImprovedStaggeredRotatingPC(const DiracImprovedStaggeredRotatingPC &dirac);
    ~DiracImprovedStaggeredRotatingPC() override;
    /** @param[in] dirac Operator state to copy; @return This operator. */
    DiracImprovedStaggeredRotatingPC &operator=(const DiracImprovedStaggeredRotatingPC &dirac);

    /**
       Refresh ordinary and rotating resident-field pointers after a gauge reload.
       @param[in] gauge Thin gauge, unused by improved staggered operators.
       @param[in] fat_gauge_in Post-epsilon fat links at this operator's precision.
       @param[in] long_gauge_in Naik links at this operator's precision.
       @param[in] clover Clover field, unused by improved staggered operators.
     */
    void updateFields(GaugeField *gauge, GaugeField *fat_gauge_in, GaugeField *long_gauge_in,
                      CloverField *clover) override;

    /** @param[out] out Dslash result. @param[in] in Input spinors. @param[in] parity Output parity. */
    void Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                QudaParity parity) const override;
    /**
       Apply preconditioned Dslash followed by the parent's xpay convention.
       @param[out] out Result.
       @param[in] in Dslash input.
       @param[in] parity Output parity.
       @param[in] x Xpay vector.
       @param[in] k Xpay scale.
     */
    void DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, QudaParity parity,
                    cvector_ref<const ColorSpinorField> &x, double k) const override;
  };
}
