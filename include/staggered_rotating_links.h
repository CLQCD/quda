#pragma once

#include <quda.h>

#ifdef __cplusplus
#include <array>
#include <cstddef>
#include <gauge_field.h>

namespace quda
{
  /**
     Named views of the four direction-packed HISQ rotating orbital/spin path fields.
     @tparam FieldPointer Const or mutable GaugeField pointer type.
   */
  template <typename FieldPointer> struct HISQRotatingOrbitalSpinLinkCacheFields {
    FieldPointer vxxtau_minus_t = nullptr; /** VXXTau paths with a negative temporal hop. */
    FieldPointer vxxtau_plus_t = nullptr;  /** VXXTau paths with a positive temporal hop. */
    FieldPointer vxyt_minus_t = nullptr;   /** VXYT paths with a negative temporal hop. */
    FieldPointer vxyt_plus_t = nullptr;    /** VXYT paths with a positive temporal hop. */

    /** @return Whether at least one path field is present. */
    bool any() const { return vxxtau_minus_t || vxxtau_plus_t || vxyt_minus_t || vxyt_plus_t; }
    /** @return Whether all four path fields are present. */
    bool complete() const { return vxxtau_minus_t && vxxtau_plus_t && vxyt_minus_t && vxyt_plus_t; }
  };

  using HISQRotatingOrbitalSpinLinkCache = HISQRotatingOrbitalSpinLinkCacheFields<const GaugeField *>;
  using MutableHISQRotatingOrbitalSpinLinkCache = HISQRotatingOrbitalSpinLinkCacheFields<GaugeField *>;

  /**
     Fixed-capacity, non-owning input view for the rotating staggered Dslash.

     A one-field view selects cache-off mode and contains the extended,
     pre-epsilon level-2 X links. A four-field view selects cache-on mode and
     stores VXXTau(-t), VXXTau(+t), VXYT(-t), and VXYT(+t), in that order.
     Constructing this view neither allocates nor copies gauge data.
   */
  struct StaggeredRotatingGaugeFieldList {
  private:
    std::array<const GaugeField *, 4> field = {}; /** Non-owning fields in the documented positional order. */
    size_t count = 0;                            /** Active entries; the only valid values are one and four. */

  public:
    StaggeredRotatingGaugeFieldList() = default;

    /** @param[in] level2_x Extended pre-epsilon level-2 X links for cache-off mode. */
    explicit StaggeredRotatingGaugeFieldList(const GaugeField &level2_x) : field {&level2_x}, count(1) { }

    /** @param[in] cache Four complete named path-cache fields for cache-on mode. */
    explicit StaggeredRotatingGaugeFieldList(const HISQRotatingOrbitalSpinLinkCache &cache) :
      field {cache.vxxtau_minus_t, cache.vxxtau_plus_t, cache.vxyt_minus_t, cache.vxyt_plus_t}, count(4)
    {
    }

    /** @return Number of active non-owning field references. */
    size_t size() const { return count; }
    /** @param[in] index Positional field index. @return Non-owning pointer, which may be null before validation. */
    const GaugeField *get(size_t index) const { return field[index]; }
    /** @param[in] index Positional field index. @return Referenced gauge field. */
    const GaugeField &operator[](size_t index) const { return *field[index]; }
    /** @return Whether the view selects the four-field cached representation. */
    bool cached() const { return count == 4; }
    /** @return The four entries decoded into their named cache roles. */
    HISQRotatingOrbitalSpinLinkCache cache() const
    {
      return count == 4 ? HISQRotatingOrbitalSpinLinkCache {field[0], field[1], field[2], field[3]}
                        : HISQRotatingOrbitalSpinLinkCache {};
    }
  };

  /** @param[in] precision Solver precision. @return Active cache-off level-2 X field, or nullptr. */
  GaugeField *getRotatingXGauge(QudaPrecision precision);

  /**
     @param[in] precision Requested solver precision.
     @return Named orbital/spin path-cache views; all members are nullptr in cache-off mode.
   */
  HISQRotatingOrbitalSpinLinkCache getHISQRotatingOrbitalSpinLinkCache(QudaPrecision precision);
}

extern "C" {
#endif

  /**
     Create an empty owner for precision-specific rotating HISQ link data.

     @return Opaque context handle; destroy it with destroyStaggeredRotatingLinkContextQuda.
   */
  void *createStaggeredRotatingLinkContextQuda(void);

  /**
     Destroy a rotating-link context and all fields it owns.

     @param[in] context Handle returned by createStaggeredRotatingLinkContextQuda; nullptr is allowed.
   */
  void destroyStaggeredRotatingLinkContextQuda(void *context);

  /**
     Select the context used by subsequently constructed rotating Dirac operators.

     The context remains owned by the caller. Activation is a lightweight
     binding needed because QUDA's ordinary fat/long gauge fields are resident
     process state; it does not copy link data.

     @param[in] context Initialized rotating-link context to activate.
   */
  void activateStaggeredRotatingLinkContextQuda(void *context);

  /**
     Load the pure level-2 HISQ X links for cache-off rotating Dslash.

     A copy is created for each distinct solver precision requested by
     QudaGaugeParam and extended with the x/y/t torus halo. This entry point is
     used only in cache-off mode, where Dslash forms every path on demand.
     Loading replaces the fields previously owned by this context.

     @param[in,out] context Rotating-link context receiving the fields.
     @param[in] level2_x_link Four direction pointers to the external level-2 X field.
     @param[in] param Gauge geometry, external order, location, and requested solver precisions.
   */
  void loadRotatingXGaugeQuda(void *context, void *level2_x_link, QudaGaugeParam *param);

  /**
     Build and retain the 16 HISQ rotating orbital/spin path matrices.

     Cache-on mode uses this entry point instead of loadRotatingXGaugeQuda.
     The extended level-2 X field is temporary and is released after all
     requested precision caches have been built.

     @param[in,out] context Rotating-link context receiving the caches.
     @param[in] level2_x_link Four direction pointers to the external level-2 X field.
     @param[in] param Gauge geometry, external order, location, and requested solver precisions.
   */
  void loadHISQRotatingOrbitalSpinLinkCacheQuda(void *context, void *level2_x_link, QudaGaugeParam *param);

  /**
     Copy the precise orbital/spin path cache into external PyQUDA-order fields.

     In cache-off mode a temporary cache is built from the retained level-2 X
     field only for this export. The four outputs preserve the native
     direction-packed representation; PyQUDA exposes physical xxt, yyt, and
     xyt independent snapshot views on top of them.

     @param[out] vxxtau_minus_t Four direction pointers for VXXTau with negative temporal hop.
     @param[out] vxxtau_plus_t Four direction pointers for VXXTau with positive temporal hop.
     @param[out] vxyt_minus_t Four direction pointers for VXYT with negative temporal hop.
     @param[out] vxyt_plus_t Four direction pointers for VXYT with positive temporal hop.
     @param[in] context Initialized rotating-link context.
     @param[in] param External field order, location, precision, and lattice geometry.
   */
  void saveHISQRotatingOrbitalSpinLinkCacheQuda(void *vxxtau_minus_t, void *vxxtau_plus_t,
                                                void *vxyt_minus_t, void *vxyt_plus_t,
                                                void *context, QudaGaugeParam *param);

#ifdef __cplusplus
}
#endif
