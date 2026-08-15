#include <dirac_staggered_rotating.h>
#include <dslash_quda.h>
#include <staggered_rotating_links.h>
#include <staggered_rotating_halo.h>

namespace quda
{
  // The active context remains caller-owned. Dirac objects retain immutable
  // field pointers resolved at construction or updateFields, while each
  // operator owns only its reusable spinor halo.

  namespace
  {
    /**
       Resolve one precision's cache-on or cache-off representation.
       @param[in] precision Fat-link precision required by this Dirac operator.
       @return Non-owning one-field X-link or four-field path-cache view.
     */
    StaggeredRotatingGaugeFieldList resolve_rotation_links(QudaPrecision precision)
    {
      const GaugeField *rotating_x = getRotatingXGauge(precision);
      const auto rotating_cache = getHISQRotatingOrbitalSpinLinkCache(precision);
      if (!rotating_x && !rotating_cache.complete())
        errorQuda("Rotating staggered operator requires level-2 X links or an orbital/spin cache at precision %d",
                  precision);
      if (rotating_x && rotating_cache.any())
        errorQuda("Rotating staggered operator found both level-2 X links and an orbital/spin cache");
      if (rotating_x && rotating_x->Precision() != precision)
        errorQuda("Rotating fat-link precision %d does not match HISQ fat-link precision %d",
                  rotating_x->Precision(), precision);

      if (rotating_cache.any() && !rotating_cache.complete())
        errorQuda("Rotating HISQ orbital/spin link cache is incomplete");
      if (!rotating_cache.complete()) return StaggeredRotatingGaugeFieldList {*rotating_x};

      auto check_precision = [precision](const GaugeField *field, const char *name) {
        if (field->Precision() != precision)
          errorQuda("Rotating %s cache precision %d does not match fat-link precision %d", name,
                    field->Precision(), precision);
      };
      check_precision(rotating_cache.vxxtau_minus_t, "VXXTau(-t)");
      check_precision(rotating_cache.vxxtau_plus_t, "VXXTau(+t)");
      check_precision(rotating_cache.vxyt_minus_t, "VXYT(-t)");
      check_precision(rotating_cache.vxyt_plus_t, "VXYT(+t)");
      return StaggeredRotatingGaugeFieldList {rotating_cache};
    }

    /**
       Validate the fixed shift-center-half geometry at setup boundaries.
       @param[in] field Local gauge geometry used to recover global extents.
     */
    void validate_rotation_geometry(const GaugeField &field)
    {
      for (int d = 0; d < 2; d++) {
        const int global_dim = field.X()[d] * comm_dim(d);
        if (global_dim % 2 != 0)
          errorQuda("Rotating staggered operator requires an even global extent in direction %d, got %d",
                    d, global_dim);
      }
    }
  }

  DiracImprovedStaggeredRotating::DiracImprovedStaggeredRotating(const DiracParam &param) :
    DiracImprovedStaggered(param),
    angular_velocity(param.angular_velocity),
    rotating_fields(resolve_rotation_links(param.fatGauge->Precision()))
  {
    validate_rotation_geometry(*param.fatGauge);
  }

  DiracImprovedStaggeredRotating::DiracImprovedStaggeredRotating(
    const DiracImprovedStaggeredRotating &dirac) :
    DiracImprovedStaggered(dirac),
    angular_velocity(dirac.angular_velocity),
    rotating_fields(dirac.rotating_fields)
  { }

  DiracImprovedStaggeredRotating::~DiracImprovedStaggeredRotating() = default;

  DiracImprovedStaggeredRotating &DiracImprovedStaggeredRotating::operator=(
    const DiracImprovedStaggeredRotating &dirac)
  {
    if (&dirac != this) {
      DiracImprovedStaggered::operator=(dirac);
      angular_velocity = dirac.angular_velocity;
      rotating_fields = dirac.rotating_fields;
      spinor_halo.reset();
    }
    return *this;
  }

  void DiracImprovedStaggeredRotating::updateFields(GaugeField *, GaugeField *fat_gauge_in,
                                                    GaugeField *long_gauge_in, CloverField *)
  {
    if (!fat_gauge_in) errorQuda("Rotating staggered update requires fat links");
    DiracImprovedStaggered::updateFields(nullptr, fat_gauge_in, long_gauge_in, nullptr);
    validate_rotation_geometry(*fat_gauge_in);
    rotating_fields = resolve_rotation_links(fat_gauge_in->Precision());
    spinor_halo.reset();
  }

  void DiracImprovedStaggeredRotating::Dslash(cvector_ref<ColorSpinorField> &out,
                                               cvector_ref<const ColorSpinorField> &in,
                                               QudaParity parity) const
  {
    if (!spinor_halo) spinor_halo = std::make_unique<StaggeredRotatingHalo>();
    DiracImprovedStaggered::Dslash(out, in, parity);
    ApplyStaggeredRotating(out, in, rotating_fields, parity, dagger, angular_velocity, commDim.data,
                           spinor_halo.get());
  }

  void DiracImprovedStaggeredRotating::DslashXpay(cvector_ref<ColorSpinorField> &out,
                                                   cvector_ref<const ColorSpinorField> &in, QudaParity parity,
                                                   cvector_ref<const ColorSpinorField> &x, double k) const
  {
    if (!spinor_halo) spinor_halo = std::make_unique<StaggeredRotatingHalo>();
    DiracImprovedStaggered::DslashXpay(out, in, parity, x, k);
    ApplyStaggeredRotating(out, in, rotating_fields, parity, dagger == QUDA_DAG_NO, angular_velocity,
                           commDim.data, spinor_halo.get());
  }

  void DiracImprovedStaggeredRotating::M(cvector_ref<ColorSpinorField> &out,
                                          cvector_ref<const ColorSpinorField> &in) const
  {
    if (!spinor_halo) spinor_halo = std::make_unique<StaggeredRotatingHalo>();
    DiracImprovedStaggered::M(out, in);
    ApplyStaggeredRotating(out, in, rotating_fields, QUDA_INVALID_PARITY, dagger == QUDA_DAG_NO,
                           angular_velocity, commDim.data, spinor_halo.get());
  }

  DiracImprovedStaggeredRotatingPC::DiracImprovedStaggeredRotatingPC(const DiracParam &param) :
    DiracImprovedStaggeredPC(param),
    angular_velocity(param.angular_velocity),
    rotating_fields(resolve_rotation_links(param.fatGauge->Precision()))
  {
    validate_rotation_geometry(*param.fatGauge);
  }

  DiracImprovedStaggeredRotatingPC::DiracImprovedStaggeredRotatingPC(
    const DiracImprovedStaggeredRotatingPC &dirac) :
    DiracImprovedStaggeredPC(dirac),
    angular_velocity(dirac.angular_velocity),
    rotating_fields(dirac.rotating_fields)
  { }

  DiracImprovedStaggeredRotatingPC::~DiracImprovedStaggeredRotatingPC() = default;

  DiracImprovedStaggeredRotatingPC &DiracImprovedStaggeredRotatingPC::operator=(
    const DiracImprovedStaggeredRotatingPC &dirac)
  {
    if (&dirac != this) {
      DiracImprovedStaggeredPC::operator=(dirac);
      angular_velocity = dirac.angular_velocity;
      rotating_fields = dirac.rotating_fields;
      spinor_halo.reset();
    }
    return *this;
  }

  void DiracImprovedStaggeredRotatingPC::updateFields(GaugeField *, GaugeField *fat_gauge_in,
                                                      GaugeField *long_gauge_in, CloverField *)
  {
    if (!fat_gauge_in) errorQuda("Rotating staggered update requires fat links");
    DiracImprovedStaggeredPC::updateFields(nullptr, fat_gauge_in, long_gauge_in, nullptr);
    validate_rotation_geometry(*fat_gauge_in);
    rotating_fields = resolve_rotation_links(fat_gauge_in->Precision());
    spinor_halo.reset();
  }

  void DiracImprovedStaggeredRotatingPC::Dslash(cvector_ref<ColorSpinorField> &out,
                                                 cvector_ref<const ColorSpinorField> &in,
                                                 QudaParity parity) const
  {
    if (!spinor_halo) spinor_halo = std::make_unique<StaggeredRotatingHalo>();
    DiracImprovedStaggeredPC::Dslash(out, in, parity);
    ApplyStaggeredRotating(out, in, rotating_fields, parity, dagger, angular_velocity, commDim.data,
                           spinor_halo.get());
  }

  void DiracImprovedStaggeredRotatingPC::DslashXpay(cvector_ref<ColorSpinorField> &out,
                                                     cvector_ref<const ColorSpinorField> &in, QudaParity parity,
                                                     cvector_ref<const ColorSpinorField> &x, double k) const
  {
    if (!spinor_halo) spinor_halo = std::make_unique<StaggeredRotatingHalo>();
    DiracImprovedStaggeredPC::DslashXpay(out, in, parity, x, k);
    ApplyStaggeredRotating(out, in, rotating_fields, parity, dagger == QUDA_DAG_NO, angular_velocity,
                           commDim.data, spinor_halo.get());
  }
}
