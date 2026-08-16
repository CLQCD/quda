#include <staggered_rotating_links.h>
#include <dirac_staggered_rotating.h>
#include <comm_quda.h>

#include <array>
#include <map>
#include <memory>

namespace quda
{
  namespace
  {
    struct RotatingLinkSet {
      std::unique_ptr<GaugeField> level2_x;       /** Owned extended X links in cache-off mode. */
      std::unique_ptr<GaugeField> vxxtau_minus_t; /** Owned direction-packed VXXTau(-t) cache. */
      std::unique_ptr<GaugeField> vxxtau_plus_t;  /** Owned direction-packed VXXTau(+t) cache. */
      std::unique_ptr<GaugeField> vxyt_minus_t;   /** Owned direction-packed VXYT(-t) cache. */
      std::unique_ptr<GaugeField> vxyt_plus_t;    /** Owned direction-packed VXYT(+t) cache. */
    };

    /**
       Allocate and build one precision's direction-packed orbital/spin cache.
       @param[in] level2_x Extended level-2 X field used to form all paths.
       @return Cache storage with all four path fields populated.
     */
    RotatingLinkSet buildOrbitalSpinCache(const GaugeField &level2_x)
    {
      GaugeFieldParam cache_param(level2_x);
      cache_param.create = QUDA_NULL_FIELD_CREATE;
      cache_param.geometry = QUDA_VECTOR_GEOMETRY;
      cache_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      cache_param.nFace = 0;
      cache_param.pad = 0;
      cache_param.link_type = QUDA_GENERAL_LINKS;
      cache_param.reconstruct = QUDA_RECONSTRUCT_NO;
      for (int d = 0; d < 4; d++) {
        cache_param.x[d] -= 2 * cache_param.r[d];
        cache_param.r[d] = 0;
      }

      RotatingLinkSet set;
      set.vxxtau_minus_t = std::make_unique<GaugeField>(cache_param);
      set.vxxtau_plus_t = std::make_unique<GaugeField>(cache_param);
      set.vxyt_minus_t = std::make_unique<GaugeField>(cache_param);
      set.vxyt_plus_t = std::make_unique<GaugeField>(cache_param);
      MutableHISQRotatingOrbitalSpinLinkCache cache {
        set.vxxtau_minus_t.get(), set.vxxtau_plus_t.get(), set.vxyt_minus_t.get(), set.vxyt_plus_t.get()};
      BuildHISQRotatingOrbitalSpinLinkCache(cache, level2_x);
      return set;
    }

    /**
       Wrap an external direction-pointer array and copy one native cache field.
       @param[out] output External cache destination.
       @param[in] source Native direction-packed cache field.
       @param[in] param External order, location, precision, and lattice geometry.
     */
    void saveCacheField(void *output, const GaugeField &source, QudaGaugeParam *param)
    {
      GaugeFieldParam output_param(*param, output, QUDA_GENERAL_LINKS);
      output_param.geometry = QUDA_VECTOR_GEOMETRY;
      output_param.reconstruct = QUDA_RECONSTRUCT_NO;
      output_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      output_param.nFace = 0;
      output_param.pad = 0;
      GaugeField external(output_param);
      external.copy(source);
    }
  } // namespace

  /** Owner for one rotating HISQ link representation at every solver precision. */
  class StaggeredRotatingLinkContext
  {
    std::map<QudaPrecision, RotatingLinkSet> links; /** Owned rotating representation for each solver precision. */
    QudaPrecision precise_precision = QUDA_INVALID_PRECISION; /** Precision exported to user-visible snapshots. */

    /**
       @param[in] precision Solver precision to locate.
       @return Matching link set, or nullptr when it is absent.
     */
    RotatingLinkSet *find(QudaPrecision precision)
    {
      auto entry = links.find(precision);
      return entry == links.end() ? nullptr : &entry->second;
    }

    /** Const overload of find(). */
    const RotatingLinkSet *find(QudaPrecision precision) const
    {
      auto entry = links.find(precision);
      return entry == links.end() ? nullptr : &entry->second;
    }

  public:
    /** @return Whether this context contains a loaded X field or complete cache. */
    bool initialized() const { return !links.empty(); }

    /**
       Replace this context's fields from an external level-2 X field.
       @param[in] level2_x_link External four-direction level-2 X field.
       @param[in] param Input geometry/order and requested solver precisions.
       @param[in] cache_links Whether to retain path caches instead of extended X fields.
     */
    void load(void *level2_x_link, QudaGaugeParam *param, bool cache_links)
    {
      if (param->type != QUDA_ASQTAD_FAT_LINKS)
        errorQuda("Rotating level-2 X links require QUDA_ASQTAD_FAT_LINKS, got %d", param->type);
      if (param->reconstruct != QUDA_RECONSTRUCT_NO)
        errorQuda("Rotating level-2 X links require QUDA_RECONSTRUCT_NO, got %d", param->reconstruct);
      for (int d = 0; d < 4; d++)
        if (param->X[d] <= 0) errorQuda("Rotating level-2 X links require positive lattice extent X[%d]", d);

      const std::array<QudaPrecision, 5> requested_precision {
        param->cuda_prec, param->cuda_prec_sloppy, param->cuda_prec_refinement_sloppy,
        param->cuda_prec_precondition, param->cuda_prec_eigensolver};
      for (auto precision : requested_precision)
        if (precision != QUDA_DOUBLE_PRECISION && precision != QUDA_SINGLE_PRECISION)
          errorQuda("Rotating staggered Dslash supports double and single precision, got %d", precision);

      GaugeFieldParam source_param(*param, level2_x_link);
      if (source_param.order <= 4) source_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      std::unique_ptr<GaugeField> source(GaugeField::Create(source_param));

      links.clear();
      precise_precision = param->cuda_prec;
      for (auto precision : requested_precision) {
        if (links.count(precision)) continue;

        GaugeFieldParam device_param(*param);
        device_param.location = QUDA_CUDA_FIELD_LOCATION;
        device_param.create = QUDA_NULL_FIELD_CREATE;
        device_param.reconstruct = QUDA_RECONSTRUCT_NO;
        device_param.setPrecision(precision, true);
        device_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;

        GaugeField local_x(device_param);
        local_x.copy(*source);

        // The paths reach two sites in x/y and one in t. A one-site z pad is
        // needed only to satisfy the generic extended exchange on a z split.
        const lat_dim_t radius = {2, 2, comm_dim_partitioned(2) ? 1 : 0, 1};
        std::unique_ptr<GaugeField> extended_x(createExtendedGauge(local_x, radius, getProfile(), true));
        if (cache_links) {
          // The temporary extended X field is released after cache construction.
          links.emplace(precision, buildOrbitalSpinCache(*extended_x));
        } else {
          RotatingLinkSet set;
          set.level2_x = std::move(extended_x);
          links.emplace(precision, std::move(set));
        }
      }
    }

    /** @param[in] precision Solver precision. @return Retained cache-off X field, or nullptr. */
    GaugeField *level2X(QudaPrecision precision) const
    {
      auto *set = find(precision);
      return set ? set->level2_x.get() : nullptr;
    }

    /**
       @param[in] precision Solver precision.
       @return Direction-packed orbital/spin cache views, or an empty view in cache-off mode.
     */
    HISQRotatingOrbitalSpinLinkCache cache(QudaPrecision precision) const
    {
      auto *set = find(precision);
      return set ? HISQRotatingOrbitalSpinLinkCache {set->vxxtau_minus_t.get(), set->vxxtau_plus_t.get(),
                                                     set->vxyt_minus_t.get(), set->vxyt_plus_t.get()}
                 : HISQRotatingOrbitalSpinLinkCache {};
    }

    /** @return Precision used for user-visible cache snapshots. */
    QudaPrecision precisePrecision() const { return precise_precision; }
  };

  namespace
  {
    StaggeredRotatingLinkContext *active_context = nullptr; /** Non-owning process-wide Dirac construction source. */

    /** @param[in] context Opaque context handle. @return Checked C++ context reference. */
    StaggeredRotatingLinkContext &getContext(void *context)
    {
      if (!context) errorQuda("Rotating link context is null");
      return *static_cast<StaggeredRotatingLinkContext *>(context);
    }
  } // namespace

  GaugeField *getRotatingXGauge(QudaPrecision precision)
  {
    return active_context ? active_context->level2X(precision) : nullptr;
  }

  HISQRotatingOrbitalSpinLinkCache getHISQRotatingOrbitalSpinLinkCache(QudaPrecision precision)
  {
    return active_context ? active_context->cache(precision) : HISQRotatingOrbitalSpinLinkCache {};
  }
} // namespace quda

extern "C" void *createStaggeredRotatingLinkContextQuda(void)
{
  return new quda::StaggeredRotatingLinkContext;
}

extern "C" void destroyStaggeredRotatingLinkContextQuda(void *context)
{
  using namespace quda;
  if (active_context == context) active_context = nullptr;
  delete static_cast<StaggeredRotatingLinkContext *>(context);
}

extern "C" void activateStaggeredRotatingLinkContextQuda(void *context)
{
  using namespace quda;
  auto &links = getContext(context);
  if (!links.initialized()) errorQuda("Cannot activate an empty rotating link context");
  active_context = &links;
}

extern "C" void loadRotatingXGaugeQuda(void *context, void *level2_x_link, QudaGaugeParam *param)
{
  quda::getContext(context).load(level2_x_link, param, false);
}

extern "C" void loadHISQRotatingOrbitalSpinLinkCacheQuda(void *context, void *level2_x_link,
                                                          QudaGaugeParam *param)
{
  quda::getContext(context).load(level2_x_link, param, true);
}

extern "C" void saveHISQRotatingOrbitalSpinLinkCacheQuda(void *vxxtau_minus_t, void *vxxtau_plus_t,
                                                          void *vxyt_minus_t, void *vxyt_plus_t,
                                                          void *context, QudaGaugeParam *param)
{
  using namespace quda;
  auto &links = getContext(context);
  const auto precision = links.precisePrecision();
  auto cache = links.cache(precision);
  RotatingLinkSet temporary;
  if (!cache.complete()) {
    auto *level2_x = links.level2X(precision);
    if (!level2_x) errorQuda("Rotating link context has neither a complete cache nor level-2 X links");
    temporary = buildOrbitalSpinCache(*level2_x);
    cache = {temporary.vxxtau_minus_t.get(), temporary.vxxtau_plus_t.get(),
             temporary.vxyt_minus_t.get(), temporary.vxyt_plus_t.get()};
  }

  saveCacheField(vxxtau_minus_t, *cache.vxxtau_minus_t, param);
  saveCacheField(vxxtau_plus_t, *cache.vxxtau_plus_t, param);
  saveCacheField(vxyt_minus_t, *cache.vxyt_minus_t, param);
  saveCacheField(vxyt_plus_t, *cache.vxyt_plus_t, param);
}
