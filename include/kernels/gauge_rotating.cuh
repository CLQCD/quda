#pragma once

#include <gauge_field_order.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <packed_array.h>
#include <array.h>
#include <kernel.h>
#include <reduce_helper.h>
#include <reduction_kernel.h>
#include <gauge_path_helper.cuh>
#include <comm_quda.h>
#include <quda_internal.h>
#include <quda_ptr.h>

// Internal path kernel for the rotating tree-improved gauge action.  Its loop
// coefficients select a small device-resident coordinate basis because the
// direction- and site-dependent terms are not expressible with QUDA's
// symmetric, direction-shared `paths<>`.
//
// The rotating gauge context initializes rank-aware global coordinates once. Each
// path stores only a basis index and loop-anchor offset, avoiding per-force
// host-to-device copies and hundreds of duplicated volume fields.  No generic
// public scalar/path interface is exposed by this implementation.

namespace quda {

  /**
     Device-resident coordinate basis for site-dependent rotating gauge paths.

     The component-major checkerboard allocation includes a torus halo. The
     halo is initialized analytically from rank-aware global coordinates, so
     it requires no communication and no per-force host upload.
   */
  class RotatingScalarField
  {
    quda_ptr storage;      /** Owned component-major device allocation. */
    lat_dim_t x = {};      /** Local physical extents in x/y/z/t order. */
    lat_dim_t r = {};      /** Halo radius on each side of every direction. */
    lat_dim_t e = {};      /** Extended extents, e[d] = x[d] + 2 * r[d]. */
    size_t volume_cb = 0;  /** Half of the extended four-dimensional volume. */
    int n_component = 0;   /** Number of coordinate-polynomial components. */

  public:
    /**
       @param[in] local_dim Local physical extents in x/y/z/t order.
       @param[in] radius Coordinate-field halo radius in each direction.
       @param[in] n_component Number of coordinate-basis components.
     */
    RotatingScalarField(const int local_dim[4], const int radius[4], int n_component);
    RotatingScalarField(const RotatingScalarField &) = delete;
    RotatingScalarField &operator=(const RotatingScalarField &) = delete;
    RotatingScalarField(RotatingScalarField &&) = default;
    RotatingScalarField &operator=(RotatingScalarField &&) = default;

    /** @return Mutable device pointer to component-major checkerboard data. */
    double *data() { return static_cast<double *>(storage.data()); }
    /** @return Const device pointer to component-major checkerboard data. */
    const double *data() const { return static_cast<const double *>(storage.data()); }
    /** @return Local physical lattice dimensions. */
    const lat_dim_t &X() const { return x; }
    /** @return Coordinate-field halo radii. */
    const lat_dim_t &R() const { return r; }
    /** @return Extended lattice dimensions. */
    const lat_dim_t &E() const { return e; }
    /** @return Extended checkerboard volume. */
    size_t VolumeCB() const { return volume_cb; }
    /** @return Allocated device bytes. */
    size_t Bytes() const { return static_cast<size_t>(n_component) * 2 * volume_cb * sizeof(double); }
    /** @return Number of coordinate-basis components. */
    int Components() const { return n_component; }

    /** @param[in] coordinate_type Basis type for each allocated component. */
    void initializeCoordinates(const int *coordinate_type);
  };

  /** Coordinate polynomials required by the rotating tree-improved action. */
  enum RotatingScalarCoordinateType {
    ROTATING_SCALAR_CONSTANT = 0,
    ROTATING_SCALAR_X = 1,
    ROTATING_SCALAR_Y = 2,
    ROTATING_SCALAR_X2 = 3,
    ROTATING_SCALAR_Y2 = 4,
    ROTATING_SCALAR_R2 = 5,
    ROTATING_SCALAR_XY = 6,
  };

  constexpr int rotating_scalar_max_components = 6;

  /** Kernel arguments for rank-aware shift-center-half coordinate fields. */
  struct RotatingScalarCoordinateArg : kernel_param<> {
    double *field; /** Non-owning component-major checkerboard destination on device. */
    int X[4];      /** Local physical extents in x/y/z/t order. */
    int R[4];      /** Analytically populated torus-halo radii. */
    int E[4];      /** Extended extents used to decode checkerboard sites. */
    int global_X[4];      /** Global physical extents in x/y/z/t order. */
    int global_offset[4]; /** This rank's global physical origin. */
    int center[2];        /** Derived global x/y midpoint for shift-center-half coordinates. */
    int n_component;      /** Active coordinate-polynomial component count. */
    size_t volume_cb;     /** Half of the extended four-volume. */
    array<int, rotating_scalar_max_components> coordinate_type; /** Polynomial enum for each component. */

    /**
       @param[out] field Coordinate-basis destination.
       @param[in] coordinate_type_ Basis type for each component.
     */
    RotatingScalarCoordinateArg(RotatingScalarField &field, const int *coordinate_type_) :
      kernel_param(dim3(field.VolumeCB(), 2, 1)),
      field(field.data()),
      n_component(field.Components()),
      volume_cb(field.VolumeCB())
    {
      for (int d = 0; d < 4; d++) {
        X[d] = field.X()[d];
        R[d] = field.R()[d];
        E[d] = field.E()[d];
        global_X[d] = X[d] * comm_dim(d);
        global_offset[d] = X[d] * comm_coord(d);
      }
      center[0] = global_X[0] / 2;
      center[1] = global_X[1] / 2;
      for (int i = 0; i < n_component; i++) coordinate_type[i] = coordinate_type_[i];
    }
  };

  template <typename Arg> struct RotatingScalarCoordinate {
    const Arg &arg;
    /** @param[in] arg Coordinate destination and rank-aware geometry. */
    constexpr RotatingScalarCoordinate(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    /**
       Evaluate one coordinate-basis component.
       @param[in] type RotatingScalarCoordinateType value.
       @param[in] fx Shift-center-half global x coordinate.
       @param[in] fy Shift-center-half global y coordinate.
       @return Requested coordinate polynomial.
     */
    __device__ __host__ double weight(int type, double fx, double fy) const
    {
      switch (type) {
      case ROTATING_SCALAR_CONSTANT: return 1.0;
      case ROTATING_SCALAR_X: return fx;
      case ROTATING_SCALAR_Y: return fy;
      case ROTATING_SCALAR_X2: return fx * fx;
      case ROTATING_SCALAR_Y2: return fy * fy;
      case ROTATING_SCALAR_R2: return fx * fx + fy * fy;
      case ROTATING_SCALAR_XY: return fx * fy;
      default: return 0.0;
      }
    }

    /** @param[in] x_cb Extended checkerboard site. @param[in] parity Site parity. */
    __device__ __host__ void operator()(int x_cb, int parity)
    {
      int x[4];
      getCoords(x, x_cb, arg.E, parity);

      int global[4];
      for (int d = 0; d < 4; d++) {
        int coordinate = arg.global_offset[d] + x[d] - arg.R[d];
        coordinate %= arg.global_X[d];
        if (coordinate < 0) coordinate += arg.global_X[d];
        global[d] = coordinate;
      }

      const double fx = static_cast<double>(global[0] - arg.center[0]) + 0.5;
      const double fy = static_cast<double>(global[1] - arg.center[1]) + 0.5;
      for (int component = 0; component < arg.n_component; component++)
        arg.field[((size_t)component * 2 + parity) * arg.volume_cb + x_cb]
          = weight(arg.coordinate_type[component], fx, fy);
    }
  };

  /**
     Device path metadata with optional site-dependent coordinate fields.

     input_path, length, path_coeff, and field_index are indexed first by
     path-table dimension. field_offset stores one x/y/z/t loop-anchor offset
     per path.
   */
  template <int dim_>
  struct paths_sitewise {
    static constexpr int dim = dim_;
    const int num_paths;       /** Number of paths in each table dimension. */
    const int max_length;      /** Direction-code stride allocated for every path. */
    int *input_path[dim];      /** Non-owning slices of buffer with direct QUDA path codes. */
    int *length[dim];          /** Non-owning slices with active path lengths. */
    double *path_coeff[dim];   /** Non-owning slices with dimensionless path multipliers. */
    int *field_index[dim];     /** Coordinate-basis component per path, or -1 for constant one. */
    int *field_offset[dim];    /** Loop-anchor offsets packed as [path][x/y/z/t]. */
    int *buffer;               /** Owned contiguous device metadata; released explicitly by free(). */

    /**
       Upload path metadata into one contiguous device allocation.
       @param[in] input_path_h Host signed-direction path tables.
       @param[in] length_h Length of each path.
       @param[in] path_coeff_h Constant multiplier of each path.
       @param[in] field_index_h Coordinate-basis component per path, or -1.
       @param[in] field_offset_h Coordinate-sampling anchor offset per path.
       @param[in] num_paths Number of allocated paths in each table dimension.
       @param[in] max_length Allocated direction-code stride per path.
     */
    paths_sitewise(std::vector<int **> &input_path_h, std::vector<std::vector<int>> &length_h,
                   std::vector<std::vector<double>> &path_coeff_h, std::vector<std::vector<int>> &field_index_h,
                   std::vector<std::vector<array<int, 4>>> &field_offset_h, int num_paths, int max_length) :
      num_paths(num_paths),
      max_length(max_length)
    {
      if (static_cast<int>(input_path_h.size()) != dim) errorQuda("input_path size %lu != %d", input_path_h.size(), dim);

      // layout in one device buffer: [dim][num_paths][max_length] int (paths)
      //   then [dim][num_paths] int (length), [dim][num_paths] int
      //   (field_index), [dim][num_paths][4] int (field_offset),
      //   pad, [dim][num_paths] double (path_coeff)
      size_t n_int = (size_t)dim * num_paths * max_length + (size_t)dim * num_paths * 6;
      size_t bytes = n_int * sizeof(int);
      int pad = ((sizeof(double) - bytes % sizeof(double)) % sizeof(double)) / sizeof(int);
      bytes += pad * sizeof(int) + (size_t)dim * num_paths * sizeof(double);

      buffer = static_cast<int *>(pool_device_malloc(bytes));
      int *h = static_cast<int *>(safe_malloc(bytes));
      memset(h, 0, bytes);

      // paths
      for (int dir = 0; dir < dim; dir++)
        for (int i = 0; i < num_paths; i++)
          for (int j = 0; j < length_h[dir][i]; j++)
            h[dir * num_paths * max_length + i * max_length + j] = input_path_h[dir][i][j];

      size_t off = (size_t)dim * num_paths * max_length; // int offset
      // length[dim][num_paths]
      for (int dir = 0; dir < dim; dir++)
        for (int i = 0; i < num_paths; i++) h[off + dir * num_paths + i] = length_h[dir][i];
      size_t off_len = off;
      off += (size_t)dim * num_paths;
      // field_index[dim][num_paths]
      for (int dir = 0; dir < dim; dir++)
        for (int i = 0; i < num_paths; i++) h[off + dir * num_paths + i] = field_index_h[dir][i];
      size_t off_fi = off;
      off += (size_t)dim * num_paths;
      // field_offset[dim][num_paths][4]
      for (int dir = 0; dir < dim; dir++)
        for (int i = 0; i < num_paths; i++)
          for (int d = 0; d < 4; d++)
            h[off + (dir * num_paths + i) * 4 + d] = field_offset_h[dir][i][d];
      size_t off_fo = off;
      off += (size_t)dim * num_paths * 4;
      // path_coeff[dim][num_paths] (double), after pad
      size_t off_pc = off + pad; // int units
      double *hpc = reinterpret_cast<double *>(h + off_pc);
      for (int dir = 0; dir < dim; dir++)
        for (int i = 0; i < num_paths; i++) hpc[dir * num_paths + i] = path_coeff_h[dir][i];

      qudaMemcpy(buffer, h, bytes, qudaMemcpyHostToDevice);
      host_free(h);

      for (int d = 0; d < dim; d++) {
        input_path[d] = buffer + d * num_paths * max_length;
        length[d] = buffer + off_len + d * num_paths;
        field_index[d] = buffer + off_fi + d * num_paths;
        field_offset[d] = buffer + off_fo + d * num_paths * 4;
        path_coeff[d] = reinterpret_cast<double *>(buffer + off_pc) + d * num_paths;
      }
    }

    /** Release the contiguous device metadata allocation. */
    void free() { pool_device_free(buffer); }
  };

  template <typename store_t, int nColor_, QudaReconstructType recon_u, QudaReconstructType recon_m, bool force_>
  struct GaugeForceSitewiseArg : kernel_param<> {
    using real = typename mapper<store_t>::type;
    static constexpr int nColor = nColor_;
    static constexpr bool compute_force = force_;
    using Link = Matrix<complex<real>, nColor>;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    using Gauge = typename gauge_mapper<real, recon_u>::type;
    using Mom = typename gauge_mapper<real, recon_m>::type;

    Mom mom;       /** Non-owning local momentum or unprojected-force accessor. */
    const Gauge u; /** Non-owning extended, unphased thin-gauge accessor. */
    int X[4];      /** Local physical momentum extents. */
    int E[4];      /** Extended gauge extents. */
    int border[4]; /** Gauge halo offset from a physical site to its extended coordinate. */
    real epsilon;  /** Overall force scale in the caller's momentum convention. */
    const paths_sitewise<4> p; /** Non-owning copy of per-direction staple metadata views. */
    const double *coeff_field; /** Non-owning component-major coordinate-basis data. */
    int coeff_E[4];            /** Extended coordinate-field extents. */
    int coeff_R[4];            /** Coordinate-field halo radii. */
    size_t coeff_volume_cb;    /** Half of the extended coordinate-field volume. */

    /**
       @param[in,out] mom Momentum or unprojected force destination.
       @param[in] u Extended thin gauge field.
       @param[in] epsilon Force scale.
       @param[in] p Per-link-direction staple metadata.
       @param[in] scalar Device coordinate basis.
     */
    GaugeForceSitewiseArg(GaugeField &mom, const GaugeField &u, double epsilon, const paths_sitewise<4> &p,
                          const RotatingScalarField &scalar) :
      kernel_param(dim3(mom.VolumeCB(), 2, 4)), mom(mom), u(u), epsilon(epsilon), p(p),
      coeff_field(scalar.data()), coeff_volume_cb(scalar.VolumeCB())
    {
      for (int i = 0; i < 4; i++) {
        X[i] = mom.X()[i];
        E[i] = u.X()[i];
        border[i] = (E[i] - X[i]) / 2;
        coeff_E[i] = scalar.E()[i];
        coeff_R[i] = scalar.R()[i];
      }
    }
  };

  template <typename Arg> struct GaugeForceSitewise {
    const Arg &arg;
    /** @param[in] arg Gauge force fields, paths, and coordinate basis. */
    constexpr GaugeForceSitewise(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    /** @param[in] x_cb Checkerboard link base. @param[in] parity Link parity. @param[in] dir Link direction. */
    __device__ __host__ void operator()(int x_cb, int parity, int dir)
    {
      using real = typename Arg::real;
      using Link = typename Arg::Link;

      int x[4] = {0, 0, 0, 0};
      getCoords(x, x_cb, arg.X, parity);

      Link link_prod, accum;
      packed_array<int8_t, 4> dx = {};

      for (int i = 0; i < arg.p.num_paths; i++) {
        real coeff = static_cast<real>(arg.p.path_coeff[dir][i]);
        int fi = arg.p.field_index[dir][i];
        if (fi >= 0) {
          int scalar_x[4];
          int scalar_parity = 0;
          const int *offset = arg.p.field_offset[dir] + i * 4;
          for (int d = 0; d < 4; d++) {
            scalar_x[d] = x[d] + arg.coeff_R[d] - offset[d];
            scalar_parity ^= scalar_x[d] & 1;
          }
          const int scalar_x_cb = linkIndex(scalar_x, arg.coeff_E);
          coeff *= static_cast<real>(
            arg.coeff_field[((size_t)fi * 2 + scalar_parity) * arg.coeff_volume_cb + scalar_x_cb]);
        }
        if (coeff == 0) continue;

        const int *path = arg.p.input_path[dir] + i * arg.p.max_length;

        int gauge_x[4];
        for (int d = 0; d < 4; d++) gauge_x[d] = x[d] + arg.border[d];
        dx[dir]++; // gauge path starts pre-shifted to x + e_dir
        int nbr_oddbit = (parity ^ 1);

        link_prod = computeGaugePath(arg, gauge_x, nbr_oddbit, path, arg.p.length[dir][i], dx);

        accum = accum + coeff * link_prod;
      }

      for (int d = 0; d < 4; d++) x[d] += arg.border[d];
      link_prod = arg.u(dir, linkIndex(x, arg.E), parity);
      link_prod = link_prod * accum;

      Link mom = arg.mom(dir, x_cb, parity);
      if (arg.compute_force) {
        mom = mom - arg.epsilon * link_prod;
        makeAntiHerm(mom);
      } else {
        mom = mom + arg.epsilon * link_prod;
      }
      arg.mom(dir, x_cb, parity) = mom;
    }
  };

  // --------------------------------------------------------------------------
  // Site-wise loop trace: reduces  sum_x coeff_loop(x) * ReTr[loop_p](x)  per
  // path p, where coeff_loop(x) = factor * path_coeff[p] * (field_index[p]<0 ?
  // 1 : coeff_field[field_index[p]][parity][x_cb]).  Mirrors GaugeLoopTrace but
  // with an optional per-site scalar field.  Used for the on-device energy of
  // the tree-improved rotating gauge action.  dim=1 (single, direction-agnostic
  // path list), reusing the paths_sitewise container.
  /** @return Maximum paths reduced together by one site-wise loop-trace block. */
  constexpr unsigned int max_n_batch_block_loop_trace_sitewise() { return 8; }

  template <typename store_t, int nColor_, QudaReconstructType recon_>
  struct GaugeLoopTraceSitewiseArg : public ReduceArg<array<double, 2>> {
    using real = typename mapper<store_t>::type;
    using reduce_t = array<double, 2>;
    static constexpr unsigned int max_n_batch_block = max_n_batch_block_loop_trace_sitewise();
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType recon = recon_;
    using Link = Matrix<complex<real>, nColor>;
    static_assert(nColor == 3, "Only nColor=3 enabled at this time");
    using Gauge = typename gauge_mapper<real, recon>::type;

    const Gauge u;     /** Non-owning extended, unphased thin-gauge accessor. */
    const double factor; /** Common dimensionless scale applied to every loop. */
    static constexpr int nParity = 2;
    int X[4];      /** Local physical gauge extents. */
    int E[4];      /** Extended gauge extents. */
    int border[4]; /** Gauge halo offset from physical to extended coordinates. */

    const paths_sitewise<1> p; /** Non-owning copy of closed-loop metadata views. */
    const double *coeff_field; /** Non-owning component-major coordinate-basis data. */
    int coeff_E[4];            /** Extended coordinate-field extents. */
    int coeff_R[4];            /** Coordinate-field halo radii. */
    size_t coeff_volume_cb;    /** Half of the extended coordinate-field volume. */

    /**
       @param[in] u Extended thin gauge field.
       @param[in] factor Common loop-trace scale.
       @param[in] p Direction-independent closed-loop metadata.
       @param[in] scalar Device coordinate basis.
     */
    GaugeLoopTraceSitewiseArg(const GaugeField &u, double factor, const paths_sitewise<1> &p,
                              const RotatingScalarField &scalar) :
      ReduceArg<reduce_t>(dim3(u.LocalVolumeCB(), 2, p.num_paths), p.num_paths),
      u(u),
      factor(factor),
      p(p),
      coeff_field(scalar.data()),
      coeff_volume_cb(scalar.VolumeCB())
    {
      for (int dir = 0; dir < 4; dir++) {
        border[dir] = u.R()[dir];
        E[dir] = u.X()[dir];
        X[dir] = u.X()[dir] - border[dir] * 2;
        coeff_E[dir] = scalar.E()[dir];
        coeff_R[dir] = scalar.R()[dir];
      }
    }
  };

  template <typename Arg> struct GaugeLoopSitewise : plus<typename Arg::reduce_t> {
    using reduce_t = typename Arg::reduce_t;
    using plus<reduce_t>::operator();
    static constexpr int reduce_block_dim = 2; // x_cb and parity are mapped to x
    const Arg &arg;
    /** @param[in] arg Gauge field, paths, coordinate basis, and reduction buffers. */
    constexpr GaugeLoopSitewise(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    /**
       @param[in,out] value Block-reduction accumulator.
       @param[in] x_cb Checkerboard loop anchor.
       @param[in] parity Anchor parity.
       @param[in] path_id Closed-loop index.
       @return Updated reduction value.
     */
    __device__ __host__ inline reduce_t operator()(reduce_t &value, int x_cb, int parity, int path_id)
    {
      using Link = typename Arg::Link;

      reduce_t loop_trace{0, 0};

      int x[4] = {0, 0, 0, 0};
      getCoords(x, x_cb, arg.X, parity);
      for (int dr = 0; dr < 4; ++dr) x[dr] += arg.border[dr];

      packed_array<int8_t, 4> dx = {};

      double coeff_loop = arg.factor * arg.p.path_coeff[0][path_id];
      int fi = arg.p.field_index[0][path_id];
      if (fi >= 0) {
        int scalar_x[4];
        int scalar_parity = 0;
        for (int d = 0; d < 4; d++) {
          scalar_x[d] = x[d] - arg.border[d] + arg.coeff_R[d];
          scalar_parity ^= scalar_x[d] & 1;
        }
        const int scalar_x_cb = linkIndex(scalar_x, arg.coeff_E);
        coeff_loop *= arg.coeff_field[((size_t)fi * 2 + scalar_parity) * arg.coeff_volume_cb + scalar_x_cb];
      }
      if (coeff_loop == 0) return operator()(loop_trace, value);

      const int *path = arg.p.input_path[0] + path_id * arg.p.max_length;

      Link link_prod = computeGaugePath(arg, x, parity, path, arg.p.length[0][path_id], dx);

      auto trace = getTrace(link_prod);
      loop_trace[0] = coeff_loop * (trace.real() - Arg::nColor);
      loop_trace[1] = coeff_loop * trace.imag();

      return operator()(loop_trace, value);
    }
  };
}
