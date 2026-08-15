#include <tunable_nd.h>
#include <instantiate.h>
#include <dirac_staggered_rotating.h>
#include <kernels/dslash_staggered_rotating.cuh>
#include <staggered_rotating_halo.h>

// Rotating-frame correction to staggered fermions on HISQ fat links.
// A thin TunableKernel3D wrapper instantiates the device implementation over
// reconstruct types while keeping its path algebra in the kernel header.
//
// Physics (torus + shift-center): D_rot applies 8 two-hop terms per site,
// (+/-x,+/-y) times (+/-t), weighted by coordinate factor f_sigma and staggered phase.

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon, bool cache_rotation_links, bool use_spinor_halo>
  class StaggeredRotating : public TunableKernel3D
  {
    cvector_ref<ColorSpinorField> &out;             /** Non-owning output spinor batch. */
    cvector_ref<const ColorSpinorField> &in;        /** Non-owning input spinor batch. */
    StaggeredRotatingGaugeFieldList fields;         /** Non-owning X-link or named path-cache view. */
    int parity;                                     /** Output parity, or QUDA_INVALID_PARITY for a full field. */
    bool dagger;                                    /** Whether the rotating correction is daggered. */
    double angular_velocity;                        /** Rotation angular velocity in lattice units. */
    const StaggeredRotatingHalo &spinor_halo;       /** Packed off-rank endpoint workspace. */

    unsigned int minThreads() const override { return in[0].VolumeCB(); }

  public:
    /**
       @param[in] meta Gauge field used only to select the precision and reconstruction instantiation.
       @param[in] fields One extended X field or four named path-cache fields.
       @param[in,out] out Spinors containing the ordinary HISQ result.
       @param[in] in Input spinors.
       @param[in] parity Output parity, or QUDA_INVALID_PARITY for a full field.
       @param[in] dagger Whether to dagger the rotating correction.
       @param[in] angular_velocity Rotation angular velocity in lattice units.
       @param[in] spinor_halo Packed off-rank endpoint spinors.
     */
    StaggeredRotating(const GaugeField &meta, const StaggeredRotatingGaugeFieldList &fields,
                      cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, int parity,
                      bool dagger, double angular_velocity, const StaggeredRotatingHalo &spinor_halo) :
      TunableKernel3D(in[0], in.size(), 1),
      out(out),
      in(in),
      fields(fields),
      parity(parity),
      dagger(dagger),
      angular_velocity(angular_velocity),
      spinor_halo(spinor_halo)
    {
      static_cast<void>(meta);
      strcat(aux, ",rotating");
      strcat(aux, cache_rotation_links ? ",cached" : ",on_the_fly");
      if (dagger) strcat(aux, ",dagger");
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    /** @param[in] stream Device stream used for the rotating kernel launch. */
    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (parity == QUDA_INVALID_PARITY) {
        launch<StaggeredRotatingApply>(
          tp, stream, StaggeredRotatingArg<Float, nColor, recon, cache_rotation_links, use_spinor_halo, 1>(
                        out, in, fields, 0, dagger, angular_velocity, spinor_halo));
        launch<StaggeredRotatingApply>(
          tp, stream, StaggeredRotatingArg<Float, nColor, recon, cache_rotation_links, use_spinor_halo, 1>(
                        out, in, fields, 1, dagger, angular_velocity, spinor_halo));
      } else {
        launch<StaggeredRotatingApply>(
          tp, stream, StaggeredRotatingArg<Float, nColor, recon, cache_rotation_links, use_spinor_halo, 1>(
                        out, in, fields, parity, dagger, angular_velocity, spinor_halo));
      }
    }

    void preTune() override { out.backup(); }
    void postTune() override { out.restore(); }

    long long flops() const override
    {
      // The 16 path matrices cost 37728 FLOPs per output site when built in
      // place (184 SU(3) matrix products, 56 adds and 16 real scales).  The
      // remaining 1248 FLOPs apply the 16 matrices to a color vector.  MILC
      // phase sign flips are intentionally excluded from this count.
      constexpr long long path_flops = cache_rotation_links ? 0 : 37728;
      constexpr long long apply_flops = 16 * (66 + 6 + 6);
      return in[0].Volume() * (path_flops + apply_flops) * in.size();
    }

    long long bytes() const override
    {
      long long gauge_bytes = 0;
      if constexpr (cache_rotation_links) {
        for (size_t i = 0; i < fields.size(); i++) gauge_bytes += fields[i].Bytes() / fields[i].Volume();
      } else {
        const long long link_bytes = fields[0].Bytes() / fields[0].Volume() / 4;
        gauge_bytes = 184 * link_bytes;
      }
      const long long spinor_bytes = in[0].Bytes() / in[0].Volume();
      return in[0].Volume() * (gauge_bytes + 16 * spinor_bytes) * in.size()
        + 2 * out[0].Bytes() * out.size();
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  using HISQRotatingOrbitalSpinLinkCachedLocal = StaggeredRotating<Float, nColor, recon, true, false>;

  template <typename Float, int nColor, QudaReconstructType recon>
  using StaggeredRotatingOnTheFlyLocal = StaggeredRotating<Float, nColor, recon, false, false>;

  template <typename Float, int nColor, QudaReconstructType recon>
  using HISQRotatingOrbitalSpinLinkCachedHalo = StaggeredRotating<Float, nColor, recon, true, true>;

  template <typename Float, int nColor, QudaReconstructType recon>
  using StaggeredRotatingOnTheFlyHalo = StaggeredRotating<Float, nColor, recon, false, true>;

  template <typename Float, int nColor, QudaReconstructType recon>
  class HISQRotatingOrbitalSpinLinkCacheBuilder : public TunableKernel3D
  {
    const GaugeField &U; /** Extended pre-epsilon level-2 X links. */
    const MutableHISQRotatingOrbitalSpinLinkCache &cache; /** Non-owning named cache destinations. */

    unsigned int minThreads() const override { return cache.vxxtau_minus_t->VolumeCB(); }

  public:
    /** @param[in] U Extended level-2 X links. @param[out] cache Direction-packed cache destinations. */
    HISQRotatingOrbitalSpinLinkCacheBuilder(const GaugeField &U, const MutableHISQRotatingOrbitalSpinLinkCache &cache) :
      TunableKernel3D(*cache.vxxtau_minus_t, 2, 1), U(U), cache(cache)
    {
      strcat(aux, ",rotating_link_cache");
      apply(device::get_default_stream());
    }

    /** @param[in] stream Device stream used for cache construction. */
    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<HISQRotatingOrbitalSpinLinkCacheBuild>(tp, stream, HISQRotatingOrbitalSpinLinkCacheArg<Float, nColor, recon>(cache, U));
    }

    long long flops() const override { return cache.vxxtau_minus_t->Volume() * 37728; }

    long long bytes() const override
    {
      const long long link_bytes = U.Bytes() / U.Volume() / 4;
      return cache.vxxtau_minus_t->Volume() * (184 + 16) * link_bytes;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon>
  using HISQRotatingOrbitalSpinLinkCacheBuilder_ = HISQRotatingOrbitalSpinLinkCacheBuilder<Float, nColor, recon>;

  void BuildHISQRotatingOrbitalSpinLinkCache(const MutableHISQRotatingOrbitalSpinLinkCache &cache, const GaugeField &fat)
  {
    auto check_field = [&fat](GaugeField *field, const char *name) {
      if (!field) errorQuda("Rotating %s cache field is null", name);
      checkPrecision(*field, fat);
      checkLocation(*field, fat);
      if (field->Geometry() != QUDA_VECTOR_GEOMETRY)
        errorQuda("Rotating %s cache field must use vector geometry", name);
    };
    check_field(cache.vxxtau_minus_t, "VXXTau(-t)");
    check_field(cache.vxxtau_plus_t, "VXXTau(+t)");
    check_field(cache.vxyt_minus_t, "VXYT(-t)");
    check_field(cache.vxyt_plus_t, "VXYT(+t)");

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<HISQRotatingOrbitalSpinLinkCacheBuilder_>(fat, cache);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  void ApplyStaggeredRotating(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                               const StaggeredRotatingGaugeFieldList &fields, int parity, bool dagger,
                               double angular_velocity, const int *comm_override,
                               StaggeredRotatingHalo *persistent_halo)
  {
    if (out.size() != in.size()) errorQuda("out/in size mismatch: %lu vs %lu", out.size(), in.size());
    checkPrecision(out[0], in[0]);
    checkLocation(out[0], in[0]);

    for (auto i = 0u; i < out.size(); i++) {
      if (out[i].Nspin() != 1) errorQuda("Staggered fermions require Nspin=1, got %d", out[i].Nspin());
    }
    if (fields.size() != 1 && fields.size() != 4)
      errorQuda("Rotating gauge-field list must contain one X field or four cache fields, got %lu", fields.size());
    for (size_t i = 0; i < fields.size(); i++)
      if (!fields.get(i)) errorQuda("Rotating gauge-field list entry %lu is null", i);

    int physical_X[4];
    for (int d = 0; d < 4; d++) physical_X[d] = in[0].X()[d];
    if (in[0].SiteSubset() == QUDA_PARITY_SITE_SUBSET) physical_X[0] *= 2;

    const GaugeField &meta = fields[0];
    checkPrecision(in[0], meta);
    checkLocation(in[0], meta);
    if (meta.Ncolor() != 3) errorQuda("Only Ncolor=3 supported, got %d", meta.Ncolor());
    if (meta.Ndim() != 4 || meta.Geometry() != QUDA_VECTOR_GEOMETRY)
      errorQuda("Rotating gauge fields require four-dimensional vector geometry");

    const bool full_cache = fields.cached();
    if (!full_cache) {
      const int required_radius[4] = {2, 2, 0, 1};
      for (int d = 0; d < 4; d++) {
        if (meta.R()[d] < required_radius[d])
          errorQuda("Rotating level-2 X halo radius %d in direction %d is smaller than required %d",
                    meta.R()[d], d, required_radius[d]);
        if (meta.X()[d] != physical_X[d] + 2 * meta.R()[d])
          errorQuda("Rotating level-2 X extent %d in direction %d does not match physical extent %d plus halo %d",
                    meta.X()[d], d, physical_X[d], meta.R()[d]);
      }
    } else {
      for (size_t i = 0; i < fields.size(); i++) {
        const GaugeField &field = fields[i];
        checkPrecision(meta, field);
        checkLocation(meta, field);
        if (field.Reconstruct() != meta.Reconstruct() || field.Order() != meta.Order()
            || field.Geometry() != meta.Geometry() || field.Ndim() != meta.Ndim())
          errorQuda("Rotating path-cache field %lu has inconsistent reconstruction, order, or geometry", i);
        for (int d = 0; d < 4; d++) {
          if (field.X()[d] != meta.X()[d])
            errorQuda("Rotating path-cache field %lu has inconsistent extent in direction %d", i, d);
          if (field.X()[d] != physical_X[d])
            errorQuda("Rotating path-cache extent %d in direction %d does not match spinor extent %d",
                      field.X()[d], d, physical_X[d]);
        }
      }
    }

    bool use_halo = false;
    for (int d : {0, 1, 3}) {
      if (!comm_dim_partitioned(d)) continue;
      if (comm_override && !comm_override[d])
        errorQuda("Rotating staggered Dslash does not support disabled communication in partitioned direction %d", d);
      use_halo = true;
    }

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    StaggeredRotatingHalo empty_halo;
    if (!use_halo) {
      if (full_cache) {
        instantiate<HISQRotatingOrbitalSpinLinkCachedLocal>(meta, fields, out, in, parity, dagger,
                                                            angular_velocity, empty_halo);
      } else {
        instantiate<StaggeredRotatingOnTheFlyLocal>(meta, fields, out, in, parity, dagger, angular_velocity,
                                                     empty_halo);
      }
    } else {
      StaggeredRotatingHalo temporary_halo;
      auto &halo = persistent_halo ? *persistent_halo : temporary_halo;
      constexpr size_t max_halo_batch = 12;
      for (size_t offset = 0; offset < in.size(); offset += max_halo_batch) {
        const size_t end = std::min(offset + max_halo_batch, in.size());
        cvector_ref<ColorSpinorField> out_batch {out.begin() + offset, out.begin() + end};
        cvector_ref<const ColorSpinorField> in_batch {in.begin() + offset, in.begin() + end};
        const StaggeredRotatingHalo &spinor_halo = halo.exchange(in_batch, parity);
        if (full_cache) {
          instantiate<HISQRotatingOrbitalSpinLinkCachedHalo>(meta, fields, out_batch, in_batch, parity, dagger,
                                                             angular_velocity, spinor_halo);
        } else {
          instantiate<StaggeredRotatingOnTheFlyHalo>(meta, fields, out_batch, in_batch, parity, dagger,
                                                      angular_velocity, spinor_halo);
        }
      }
    }
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
