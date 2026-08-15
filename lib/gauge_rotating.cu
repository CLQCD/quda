#include <array.h>
#include <comm_quda.h>
#include <gauge_field.h>
#include <gauge_rotating.h>
#include <gauge_path_quda.h>
#include <instantiate.h>
#include <kernels/gauge_rotating.cuh>
#include <momentum.h>
#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <util_quda.h>

#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace quda
{
  // Coordinate storage, path launchers, context ownership, and public C entry
  // points intentionally share this translation unit.  They constitute one
  // rotation-specific gauge subsystem and none has an independent consumer.

  RotatingScalarField::RotatingScalarField(const int local_dim[4], const int radius[4], int n_component) :
    n_component(n_component)
  {
    if (n_component <= 0 || n_component > rotating_scalar_max_components)
      errorQuda("Rotating scalar field component count %d is outside [1,%d]", n_component,
                rotating_scalar_max_components);

    size_t volume = 1;
    for (int d = 0; d < 4; d++) {
      if (local_dim[d] <= 0 || (d == 0 && local_dim[d] % 2 != 0))
        errorQuda("Invalid local scalar-field dimension %d in direction %d", local_dim[d], d);
      if (radius[d] < 0) errorQuda("Negative scalar-field halo %d in direction %d", radius[d], d);
      x[d] = local_dim[d];
      r[d] = radius[d];
      e[d] = x[d] + 2 * r[d];
      volume *= e[d];
    }
    if (volume % 2 != 0) errorQuda("Extended scalar-field volume must be even");
    volume_cb = volume / 2;
    storage = quda_ptr(QUDA_MEMORY_DEVICE, Bytes());
  }

  namespace
  {
    template <typename Arg, template <typename> class Functor>
    class RotatingScalarApply : public TunableKernel2D
    {
      const Arg &arg; /** Non-owning coordinate-kernel arguments valid for this immediate launch. */
      size_t bytes_;  /** Number of destination bytes overwritten per launch. */

      unsigned int minThreads() const override { return arg.threads.x; }

    public:
      /**
         @param[in] arg Coordinate kernel arguments.
         @param[in] bytes Bytes written by one kernel application.
         @param[in] label Tuning-key suffix.
       */
      RotatingScalarApply(const Arg &arg, size_t bytes, const char *label) :
        TunableKernel2D(arg.threads.x, arg.threads.y, QUDA_CUDA_FIELD_LOCATION), arg(arg), bytes_(bytes)
      {
        strcat(aux, label);
        apply(device::get_default_stream());
      }

      /** @param[in] stream Device stream used for coordinate initialization. */
      void apply(const qudaStream_t &stream)
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        launch<Functor>(tp, stream, arg);
      }

      // Coordinate initialization overwrites every element deterministically,
      // so autotuning trials are idempotent and need no full-field backup.
      long long bytes() const override { return bytes_; }
    };
  } // namespace

  void RotatingScalarField::initializeCoordinates(const int *coordinate_type)
  {
    RotatingScalarCoordinateArg arg(*this, coordinate_type);
    RotatingScalarApply<RotatingScalarCoordinateArg, RotatingScalarCoordinate> apply(arg, Bytes(), ",coordinate");
  }

  template <typename Float, int nColor, QudaReconstructType recon_u, bool compute_force = true>
  class ForceGaugeSitewise : public TunableKernel3D
  {
    const GaugeField &u;                 /** Extended, unphased thin gauge. */
    GaugeField &mom;                     /** Local momentum destination updated in place. */
    double epsilon;                      /** Overall force scale. */
    const paths_sitewise<4> &p;          /** Resident per-direction staple metadata. */
    const RotatingScalarField &scalar;   /** Resident coordinate-polynomial basis. */
    unsigned int minThreads() const { return mom.VolumeCB(); }

  public:
    /**
       @param[in] u Extended thin gauge field.
       @param[in,out] mom Momentum destination.
       @param[in] epsilon Force scale.
       @param[in] p Per-direction staple metadata.
       @param[in] scalar Device coordinate basis.
     */
    ForceGaugeSitewise(const GaugeField &u, GaugeField &mom, double epsilon, const paths_sitewise<4> &p,
                       const RotatingScalarField &scalar) :
      TunableKernel3D(u, 2, 4), u(u), mom(mom), epsilon(epsilon), p(p), scalar(scalar)
    {
      strcat(aux, ",sitewise,num_paths=");
      strcat(aux, std::to_string(p.num_paths).c_str());
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    /** @param[in] stream Device stream used for the force launch. */
    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<GaugeForceSitewise>(
        tp, stream,
        GaugeForceSitewiseArg<Float, nColor, recon_u,
                              compute_force ? QUDA_RECONSTRUCT_10 : QUDA_RECONSTRUCT_NO, compute_force>(
          mom, u, epsilon, p, scalar));
    }

    void preTune() { mom.backup(); }
    void postTune() { mom.restore(); }
    long long flops() const { return (long long)p.num_paths * p.max_length * 198ll * mom.Volume() * 4; }
    long long bytes() const { return 2 * mom.Bytes() + (long long)p.num_paths * u.Bytes() * 4; }
  };

  template <typename Float, int nColor, QudaReconstructType recon_u>
  using GaugeForceSitewise_ = ForceGaugeSitewise<Float, nColor, recon_u, true>;

  /**
     @param[in,out] mom Momentum destination.
     @param[in] u Extended thin gauge field.
     @param[in] epsilon Force scale.
     @param[in] paths Per-direction staple metadata.
     @param[in] scalar Device coordinate basis.
   */
  void gaugeForceSitewise(GaugeField &mom, const GaugeField &u, double epsilon,
                          const paths_sitewise<4> &paths, const RotatingScalarField &scalar)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    checkPrecision(mom, u);
    checkLocation(mom, u);
    if (mom.Reconstruct() != QUDA_RECONSTRUCT_10) errorQuda("Reconstruction type %d not supported", mom.Reconstruct());

    instantiate<GaugeForceSitewise_>(u, mom, epsilon, paths, scalar);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

  template <typename Float, int nColor, QudaReconstructType recon>
  class GaugeActionSitewise : public TunableMultiReduction
  {
    const GaugeField &u; /** Extended, unphased thin gauge. */
    using reduce_t = array<double, 2>;
    std::vector<reduce_t> &path_sums;   /** Host-visible real/imaginary reduction for each loop. */
    const paths_sitewise<1> &p;         /** Resident closed-loop metadata. */
    const RotatingScalarField &scalar;  /** Resident coordinate-polynomial basis. */

  public:
    /**
       @param[in] u Extended thin gauge field.
       @param[out] path_sums Per-path complex reductions.
       @param[in] p Closed-loop metadata.
       @param[in] scalar Device coordinate basis.
     */
    GaugeActionSitewise(const GaugeField &u, std::vector<reduce_t> &path_sums, const paths_sitewise<1> &p,
                        const RotatingScalarField &scalar) :
      TunableMultiReduction(u, 2u, p.num_paths, 8), u(u), path_sums(path_sums), p(p), scalar(scalar)
    {
      if (p.num_paths != static_cast<int>(path_sums.size()))
        errorQuda("Path sum size %lu != number of paths %d", path_sums.size(), p.num_paths);
      strcat(aux, ",sitewise,num_paths=");
      u32toa(aux + strlen(aux), p.num_paths);
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    /** @param[in] stream Device stream used for loop-trace reduction. */
    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      GaugeLoopTraceSitewiseArg<Float, nColor, recon> arg(u, 1.0, p, scalar);
      launch<GaugeLoopSitewise>(path_sums, tp, stream, arg);
    }

    long long flops() const override
    {
      auto Nc = u.Ncolor();
      auto mat_mul_flops = 8ll * Nc * Nc * Nc - 2 * Nc * Nc;
      return ((long long)p.num_paths * p.max_length * mat_mul_flops + p.num_paths * (2 * Nc + 2)) * u.Volume();
    }
    long long bytes() const override { return (long long)p.num_paths * p.max_length * u.Bytes() / 4; }
  };

  /**
     @param[in] u Extended thin gauge field.
     @param[in] paths Closed-loop metadata.
     @param[in] scalar Device coordinate basis.
     @return Globally reduced real action contribution.
   */
  double gaugeActionSitewise(const GaugeField &u, const paths_sitewise<1> &paths,
                             const RotatingScalarField &scalar)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    std::vector<array<double, 2>> path_sums(paths.num_paths);
    instantiate<GaugeActionSitewise, ReconstructNo12>(u, path_sums, paths, scalar);

    double action = 0.0;
    for (const auto &sum : path_sums) action += sum[0];
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return action;
  }

  namespace
  {
    constexpr int coordinate_components = 6;

    /**
       @param[in,out] dx Current path displacement.
       @param[in] step Direct QUDA path code.
       @param[in] kind Diagnostic path-table name.
       @param[in] direction Differentiated-link direction for diagnostics.
       @param[in] path Path index for diagnostics.
     */
    void update_displacement(array<int, 4> &dx, int step, const char *kind, int direction, int path)
    {
      if (step >= 0 && step < 4) {
        dx[step]++;
      } else if (step >= 4 && step < 8) {
        dx[7 - step]--;
      } else {
        errorQuda("Invalid %s step %d for direction %d path %d", kind, step, direction, path);
      }
    }

    /** @param[in] dx Path displacement. @return Whether every component is zero. */
    bool is_zero(const array<int, 4> &dx)
    {
      for (int d = 0; d < 4; d++)
        if (dx[d] != 0) return false;
      return true;
    }
  }

  /**
     Immutable owner of coordinate fields and uploaded path metadata.

     Python assembles the action-specific path tables once.  The context checks
     their topology and coordinate anchors at construction, so every energy
     and force call thereafter consists only of resident-field preparation
     and device launches.
   */
  class RotatingGaugeContext
  {
    RotatingScalarField scalar;                     /** Owned resident coordinate-polynomial basis. */
    std::unique_ptr<paths_sitewise<1>> action_paths; /** Owned host view of resident closed-loop metadata. */
    std::unique_ptr<paths_sitewise<4>> force_paths;  /** Owned host view of resident force-staple metadata. */

  public:
    /**
       Validate and upload one rotating gauge action/force definition.
       @param[in] local_dim Local physical extents in x/y/z/t order.
       @param[in] radius Coordinate-field halo radii.
       @param[in] action_path Closed action paths.
       @param[in] action_length Length of each action path.
       @param[in] action_coeff Scalar coefficient of each action path.
       @param[in] action_field_index Coordinate-basis component per action path, or -1.
       @param[in] action_num_paths Number of action paths.
       @param[in] action_max_length Allocated action-path stride.
       @param[in] force_path Staple paths grouped by differentiated-link direction.
       @param[in] force_length Length of each force staple.
       @param[in] force_coeff Scalar coefficient of each force staple.
       @param[in] force_field_index Coordinate-basis component per force staple, or -1.
       @param[in] force_field_offset Loop-anchor offset per force staple.
       @param[in] force_num_paths Allocated force paths per direction.
       @param[in] force_max_length Allocated force-path stride.
     */
    RotatingGaugeContext(const int local_dim[4], const int radius[4],
                      int **action_path, const int *action_length, const double *action_coeff,
                      const int *action_field_index, int action_num_paths, int action_max_length,
                      int ***force_path, int **force_length, double **force_coeff,
                      int **force_field_index, int ***force_field_offset, int force_num_paths,
                      int force_max_length) :
      scalar(local_dim, radius, coordinate_components)
    {
      if (!action_path || !action_length || !action_coeff || !action_field_index
          || !force_path || !force_length || !force_coeff || !force_field_index || !force_field_offset)
        errorQuda("Null metadata passed to rotating gauge context");
      for (int direction = 0; direction < 4; direction++) {
        if (!force_path[direction] || !force_length[direction] || !force_coeff[direction]
            || !force_field_index[direction] || !force_field_offset[direction])
          errorQuda("Null force metadata for direction %d", direction);
      }
      if (action_num_paths <= 0 || action_max_length <= 0)
        errorQuda("Invalid rotating action path shape %d x %d", action_num_paths, action_max_length);
      if (force_num_paths <= 0 || force_max_length <= 0)
        errorQuda("Invalid rotating force path shape %d x %d", force_num_paths, force_max_length);
      for (int direction = 0; direction < 2; direction++) {
        const int global_dim = local_dim[direction] * comm_dim(direction);
        if (global_dim % 2 != 0)
          errorQuda("Rotating gauge context requires an even global extent in direction %d, got %d",
                    direction, global_dim);
      }


      const int coordinate_type[coordinate_components] = {
        ROTATING_SCALAR_X, ROTATING_SCALAR_Y, ROTATING_SCALAR_X2,
        ROTATING_SCALAR_Y2, ROTATING_SCALAR_R2, ROTATING_SCALAR_XY};
      scalar.initializeCoordinates(coordinate_type);

      std::vector<int **> action_path_v(1, action_path);
      std::vector<std::vector<int>> action_length_v(1, std::vector<int>(action_num_paths));
      std::vector<std::vector<double>> action_coeff_v(1, std::vector<double>(action_num_paths));
      std::vector<std::vector<int>> action_field_v(1, std::vector<int>(action_num_paths));
      std::vector<std::vector<array<int, 4>>> action_offset_v(
        1, std::vector<array<int, 4>>(action_num_paths));

      for (int path = 0; path < action_num_paths; path++) {
        if (!action_path[path]) errorQuda("Null rotating action path %d", path);
        const int length = action_length[path];
        if (length <= 0 || length > action_max_length)
          errorQuda("Invalid rotating action length %d for path %d", length, path);
        const int field = action_field_index[path];
        if (field < -1 || field >= coordinate_components)
          errorQuda("Invalid rotating action field %d for path %d", field, path);
        array<int, 4> dx = {};
        for (int step = 0; step < length; step++)
          update_displacement(dx, action_path[path][step], "action", 0, path);
        if (!is_zero(dx)) errorQuda("Rotating action path %d is not closed", path);

        action_length_v[0][path] = length;
        action_coeff_v[0][path] = action_coeff[path];
        action_field_v[0][path] = field;
      }

      std::vector<int **> force_path_v(4);
      std::vector<std::vector<int>> force_length_v(4, std::vector<int>(force_num_paths));
      std::vector<std::vector<double>> force_coeff_v(4, std::vector<double>(force_num_paths));
      std::vector<std::vector<int>> force_field_v(4, std::vector<int>(force_num_paths));
      std::vector<std::vector<array<int, 4>>> force_offset_v(
        4, std::vector<array<int, 4>>(force_num_paths));

      for (int direction = 0; direction < 4; direction++) {
        force_path_v[direction] = force_path[direction];
        for (int path = 0; path < force_num_paths; path++) {
          const int length = force_length[direction][path];
          const double coefficient = force_coeff[direction][path];
          const int field = force_field_index[direction][path];

          if (length == 0 && std::abs(coefficient) <= std::numeric_limits<double>::epsilon()) {
            force_length_v[direction][path] = 0;
            force_coeff_v[direction][path] = 0.0;
            force_field_v[direction][path] = -1;
            continue;
          }
          if (!force_path[direction][path] || !force_field_offset[direction][path])
            errorQuda("Null rotating force path metadata for direction %d path %d", direction, path);
          if (length <= 0 || length > force_max_length)
            errorQuda("Invalid rotating force length %d for direction %d path %d",
                      length, direction, path);
          if (field < -1 || field >= coordinate_components)
            errorQuda("Invalid rotating force field %d for direction %d path %d",
                      field, direction, path);

          array<int, 4> dx = {};
          dx[direction] = 1;
          for (int step = 0; step < length; step++)
            update_displacement(dx, force_path[direction][path][step], "force", direction, path);
          if (!is_zero(dx))
            errorQuda("Rotating force staple for direction %d path %d does not close its link",
                      direction, path);

          for (int d = 0; d < 4; d++) {
            const int offset = force_field_offset[direction][path][d];
            if (field >= 0 && (offset < -scalar.R()[d] || offset > scalar.R()[d]))
              errorQuda("Rotating force offset %d exceeds scalar halo %d in direction %d path %d axis %d",
                        offset, scalar.R()[d], direction, path, d);
            force_offset_v[direction][path][d] = offset;
          }

          force_length_v[direction][path] = length;
          force_coeff_v[direction][path] = coefficient;
          force_field_v[direction][path] = field;
        }
      }
      action_paths = std::make_unique<paths_sitewise<1>>(
        action_path_v, action_length_v, action_coeff_v, action_field_v, action_offset_v,
        action_num_paths, action_max_length);
      force_paths = std::make_unique<paths_sitewise<4>>(
        force_path_v, force_length_v, force_coeff_v, force_field_v, force_offset_v,
        force_num_paths, force_max_length);
    }

    RotatingGaugeContext(const RotatingGaugeContext &) = delete;
    RotatingGaugeContext &operator=(const RotatingGaugeContext &) = delete;

    ~RotatingGaugeContext()
    {
      if (force_paths) force_paths->free();
      if (action_paths) action_paths->free();
    }

    /** @return Device coordinate basis owned by this context. */
    const RotatingScalarField &Scalar() const { return scalar; }
    /** @return Uploaded closed action paths. */
    const paths_sitewise<1> &ActionPaths() const { return *action_paths; }
    /** @return Uploaded per-direction force staples. */
    const paths_sitewise<4> &ForcePaths() const { return *force_paths; }

    /** @param[in] gauge Gauge field whose local geometry must match this context. */
    void checkGeometry(const GaugeField &gauge) const
    {
      for (int direction = 0; direction < 4; direction++) {
        const int local_dim = gauge.X()[direction] - 2 * gauge.R()[direction];
        if (local_dim != scalar.X()[direction])
          errorQuda("Rotating gauge context local dimension %d does not match gauge dimension %d in direction %d",
                    scalar.X()[direction], local_dim, direction);
      }
    }
  };

  namespace
  {
    /** @param[in] context Opaque C handle. @return Checked rotating gauge context. */
    RotatingGaugeContext &get_context(void *context)
    {
      if (!context) errorQuda("Rotating gauge context is null");
      return *static_cast<RotatingGaugeContext *>(context);
    }
  }

  /**
     @param[in,out] mom Momentum destination.
     @param[in] u Extended, unphased thin gauge field.
     @param[in] epsilon Force scale.
     @param[in] context Rotating gauge context.
   */
  void gaugeForceRotating(GaugeField &mom, const GaugeField &u, double epsilon, void *context)
  {
    auto &rotation = get_context(context);
    rotation.checkGeometry(u);
    gaugeForceSitewise(mom, u, epsilon, rotation.ForcePaths(), rotation.Scalar());
  }

  /** @param[in] u Extended, unphased thin gauge. @param[in] context Gauge context. @return Action energy. */
  double gaugeActionRotating(const GaugeField &u, void *context)
  {
    auto &rotation = get_context(context);
    rotation.checkGeometry(u);
    return gaugeActionSitewise(u, rotation.ActionPaths(), rotation.Scalar());
  }

  GaugeField *getResidentGauge();

  int computeGaugeRotatingForce(GaugeField &resident_momentum, void *mom, void *context,
                                double dt, QudaGaugeParam *param)
  {
    if (!context) errorQuda("Rotating gauge force requires a device context");
    if (!param->use_resident_gauge)
      errorQuda("Rotating gauge force requires the resident thin gauge");

    GaugeField *resident = getResidentGauge();
    if (!resident) errorQuda("Cannot compute rotating gauge force without a resident gauge field");

    GaugeFieldParam mom_param(*param, mom, QUDA_ASQTAD_MOM_LINKS);
    mom_param.location = QUDA_CPU_FIELD_LOCATION;
    GaugeField cpu_mom = !param->use_resident_mom ? GaugeField(mom_param) : GaugeField();

    if (param->use_resident_mom && resident_momentum.empty())
      errorQuda("No resident momentum field to use");
    mom_param.location = QUDA_CUDA_FIELD_LOCATION;
    mom_param.create = param->overwrite_mom ? QUDA_ZERO_FIELD_CREATE : QUDA_COPY_FIELD_CREATE;
    mom_param.field = &cpu_mom;
    mom_param.reconstruct = QUDA_RECONSTRUCT_10;
    mom_param.setPrecision(param->cuda_prec, true);
    GaugeField cuda_mom
      = param->use_resident_mom ? resident_momentum.create_alias() : GaugeField(mom_param);
    if (param->use_resident_mom && param->overwrite_mom) cuda_mom.zero();

    // Tree-improved loops reach at most two links away.  The resident gauge
    // is read-only, so the force owns only this temporary exchanged view and
    // never participates in ordinary QUDA gauge-residency updates.
    lat_dim_t radius = {2 * comm_dim_partitioned(0), 2 * comm_dim_partitioned(1),
                        2 * comm_dim_partitioned(2), 2 * comm_dim_partitioned(3)};
    const bool needs_copy
      = resident->StaggeredPhaseApplied() || radius[0] || radius[1] || radius[2] || radius[3];
    GaugeField *force_gauge
      = needs_copy ? createExtendedGauge(*resident, radius, getProfile()) : resident;
    if (force_gauge->StaggeredPhaseApplied()) force_gauge->removeStaggeredPhase();

    if (!forceMonitor()) {
      gaugeForceRotating(cuda_mom, *force_gauge, dt, context);
    } else {
      GaugeFieldParam force_param(cuda_mom);
      force_param.create = QUDA_ZERO_FIELD_CREATE;
      GaugeField force(force_param);
      gaugeForceRotating(force, *force_gauge, 1.0, context);
      updateMomentum(cuda_mom, dt, force, "gauge-rotating");
    }

    if (needs_copy) delete force_gauge;
    if (param->return_result_mom) cpu_mom.copy(cuda_mom);

    if (param->make_resident_mom && !param->use_resident_mom)
      std::exchange(resident_momentum, cuda_mom);
    else if (!param->make_resident_mom)
      resident_momentum = GaugeField();

    return 0;
  }
}

extern "C" void *createGaugeRotatingContextQuda(
  const int local_dim[4], const int radius[4],
  int **action_path, const int *action_length, const double *action_coeff,
  const int *action_field_index, int action_num_paths, int action_max_length,
  int ***force_path, int **force_length, double **force_coeff,
  int **force_field_index, int ***force_field_offset, int force_num_paths,
  int force_max_length)
{
  if (!local_dim || !radius) errorQuda("Null geometry passed to rotating gauge context");
  return new quda::RotatingGaugeContext(
    local_dim, radius, action_path, action_length, action_coeff,
    action_field_index, action_num_paths, action_max_length, force_path,
    force_length, force_coeff, force_field_index, force_field_offset,
    force_num_paths, force_max_length);
}

extern "C" void destroyGaugeRotatingContextQuda(void *context)
{
  delete static_cast<quda::RotatingGaugeContext *>(context);
}

extern "C" double computeGaugeRotatingActionQuda(void *context)
{
  using namespace quda;
  GaugeField *resident = getResidentGauge();
  if (!resident) errorQuda("Cannot compute rotating gauge action without a resident gauge field");

  // Tree-improved loops reach at most two links away. Partitioned dimensions
  // need an exchanged halo; local torus dimensions use the path helper's wrap.
  lat_dim_t radius = {2 * comm_dim_partitioned(0), 2 * comm_dim_partitioned(1),
                      2 * comm_dim_partitioned(2), 2 * comm_dim_partitioned(3)};
  const bool needs_copy = resident->StaggeredPhaseApplied() || radius[0] || radius[1] || radius[2] || radius[3];
  GaugeField *gauge = needs_copy ? createExtendedGauge(*resident, radius, getProfile()) : resident;
  if (gauge->StaggeredPhaseApplied()) gauge->removeStaggeredPhase();

  const double action = gaugeActionRotating(*gauge, context);
  if (needs_copy) delete gauge;
  return action;
}
