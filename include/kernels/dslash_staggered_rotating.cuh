#pragma once

#include <array>
#include <color_spinor_field_order.h>
#include <comm_quda.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <index_helper.cuh>
#include <kernel.h>
#include <kernels/staggered_rotating_halo.cuh>
#include <staggered_rotating_links.h>

// Additive rotating-frame correction to the HISQ staggered operator.
// Supported conventions are a spatial torus, anti-periodic fermions in time,
// MILC staggered phases, and a rotation axis shifted by half a lattice site.
// Dimensions use QUDA order (x,y,z,t). The effective fat links arrive with
// MILC phases and the temporal fermion-boundary sign already applied; path
// loads below strip only the MILC phase.

namespace quda
{

  /** Cache-off gauge accessor state; the level-2 X field is extended. */
  template <typename Float, QudaReconstructType reconstruct_, bool cache_rotation_links_>
  struct StaggeredRotatingGaugeArg;

  template <typename Float, QudaReconstructType reconstruct_>
  struct StaggeredRotatingGaugeArg<Float, reconstruct_, false> {
    using G = typename gauge_mapper<Float, reconstruct_>::type;
    const G U; /** Extended, phased pre-epsilon level-2 X links used to build paths on demand. */
    int E[4];  /** Extended X-link dimensions in x/y/z/t order. */
    int R[4];  /** X-link halo radius on each side of every direction. */

    /** @param[in] fields One-field cache-off rotating gauge view. */
    explicit StaggeredRotatingGaugeArg(const StaggeredRotatingGaugeFieldList &fields) : U(fields[0])
    {
      for (int d = 0; d < 4; d++) {
        E[d] = fields[0].X()[d];
        R[d] = fields[0].R()[d];
      }
    }
  };

  template <typename Float, QudaReconstructType reconstruct_>
  struct StaggeredRotatingGaugeArg<Float, reconstruct_, true> {
    using G = typename gauge_mapper<Float, reconstruct_>::type;
    const G vxxtau_minus_t; /** Direction-packed VXXTau paths with a negative temporal hop. */
    const G vxxtau_plus_t;  /** Direction-packed VXXTau paths with a positive temporal hop. */
    const G vxyt_minus_t;   /** Direction-packed VXYT paths with a negative temporal hop. */
    const G vxyt_plus_t;    /** Direction-packed VXYT paths with a positive temporal hop. */
    int R[4]; /** Endpoint halo radius {2,2,0,1}; no gauge accessor uses this radius. */

    /** @param[in] fields Four-field cache-on rotating gauge view in the documented order. */
    explicit StaggeredRotatingGaugeArg(const StaggeredRotatingGaugeFieldList &fields) :
      vxxtau_minus_t(fields[0]),
      vxxtau_plus_t(fields[1]),
      vxyt_minus_t(fields[2]),
      vxyt_plus_t(fields[3])
    {
      const int stencil_radius[4] = {2, 2, 0, 1};
      for (int d = 0; d < 4; d++) R[d] = stencil_radius[d];
    }
  };

  template <typename Float, int nColor_, QudaReconstructType reconstruct_, bool cache_rotation_links_,
            bool use_spinor_halo_, int n_src_tile = 1>
  struct StaggeredRotatingArg : kernel_param<>, StaggeredRotatingGaugeArg<Float, reconstruct_, cache_rotation_links_> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false;
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, true>::type;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    using GaugeArg = StaggeredRotatingGaugeArg<Float, reconstruct, cache_rotation_links_>;

    F out[MAX_MULTI_RHS]; /** Non-owning output accessors containing the ordinary HISQ result. */
    F in[MAX_MULTI_RHS];  /** Non-owning input spinor accessors, one per right-hand side. */
    const complex<real> *halo; /** Non-owning compact remote-spinor buffer; null for a local stencil. */

    static constexpr bool cache_rotation_links = cache_rotation_links_;
    static constexpr bool use_spinor_halo = use_spinor_halo_;

    const real angular_velocity; /** Rotation angular velocity in lattice units. */
    const real dagger_scale; /** +1.0 for D_rot, -1.0 for \f$D^\dagger_{rot}\f$ */

    int X[4];           /** Local physical spinor dimensions in x/y/z/t order. */
    int partitioned[4]; /** Whether the communicator is partitioned in each direction. */
    size_t halo_offset[StaggeredRotatingHalo::region_count]; /** Packed receive offset for every x/y/t region. */
    int halo_rhs_capacity; /** Right-hand-side stride of the packed halo allocation. */
    int global_X[4];       /** Global physical dimensions in x/y/z/t order. */
    int global_offset[4];  /** This rank's global physical origin. */
    int center[2];         /** Derived global x/y midpoint for shift-center-half coordinates. */
    const int parity;      /** Output parity, zero or one. */
    const int nParity;     /** Number of checkerboards stored by each input spinor. */

    /**
       @param[in,out] out Spinors containing the ordinary HISQ Dslash result.
       @param[in] in Input spinors.
       @param[in] fields One extended X field or four named path-cache fields.
       @param[in] parity Output parity, or QUDA_INVALID_PARITY for a full field.
       @param[in] dagger Whether to dagger the rotating correction.
       @param[in] angular_velocity Rotation angular velocity in lattice units.
       @param[in] spinor_halo Packed off-rank endpoint spinors.
     */
    StaggeredRotatingArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         const StaggeredRotatingGaugeFieldList &fields, int parity, bool dagger,
                         double angular_velocity,
                         const StaggeredRotatingHalo &spinor_halo) :
      kernel_param(dim3(in[0].VolumeCB(), in.size(), 1)),
      GaugeArg(fields),
      halo(static_cast<const complex<real> *>(spinor_halo.data())),
      angular_velocity(angular_velocity),
      dagger_scale(dagger ? static_cast<real>(-1.0) : static_cast<real>(1.0)),
      halo_rhs_capacity(spinor_halo.RHSCapacity()),
      parity(parity),
      nParity(in[0].SiteSubset())
    {
      for (int i = 0; i < 4; i++) {
        X[i] = in[0].X()[i];
        partitioned[i] = comm_dim_partitioned(i);
      }
      for (int region = 0; region < StaggeredRotatingHalo::region_count; region++)
        halo_offset[region] = spinor_halo.RegionOffset()[region];
      if (nParity == 1) X[0] *= 2;
      for (int i = 0; i < 4; i++) {
        global_X[i] = X[i] * comm_dim(i);
        global_offset[i] = X[i] * comm_coord(i);
      }
      center[0] = global_X[0] / 2;
      center[1] = global_X[1] / 2;
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
      }
    }
  };
  // ---- coordinate / index helpers (full-coordinate arithmetic, torus wrap) ----

  /** @param[in] v Coordinate to wrap. @param[in] n Torus extent. @return Coordinate in [0,n). */
  __device__ __host__ inline int wrap_ax(int v, int n) { return ((v % n) + n) % n; }

  /** @param[in] c Full coordinates. @param[in] X Local extents. @return Parity of the wrapped site. */
  __device__ __host__ inline int full_parity(const int c[4], const int X[4])
  {
    return (wrap_ax(c[0], X[0]) + wrap_ax(c[1], X[1]) + wrap_ax(c[2], X[2]) + wrap_ax(c[3], X[3])) & 1;
  }

  /** @param[in] c Full coordinates. @param[in] X Local extents. @return Wrapped checkerboard index. */
  __device__ __host__ inline int full_cb_index(const int c[4], const int X[4])
  {
    int x = wrap_ax(c[0], X[0]), y = wrap_ax(c[1], X[1]);
    int z = wrap_ax(c[2], X[2]), t = wrap_ax(c[3], X[3]);
    return (((t * X[2] + z) * X[1] + y) * X[0] + x) >> 1;
  }

  /**
   * Identify an off-rank stencil endpoint.
   * @param[in] site Possibly off-rank local coordinates.
   * @param[out] wrapped Coordinates wrapped into the local torus.
   * @param[in] arg Local geometry and partitioning metadata.
   * @return Whether site must be read from the diagonal halo.
   */
  template <typename Arg>
  __device__ __host__ inline bool rotating_remote_site(const int site[4], int wrapped[4], const Arg &arg)
  {
    bool remote = false;
    for (int d = 0; d < 4; d++) {
      const bool outside = site[d] < 0 || site[d] >= arg.X[d];
      remote |= outside && arg.partitioned[d];
      wrapped[d] = wrap_ax(site[d], arg.X[d]);
    }
    return remote;
  }

  /**
   * @param[in] value Possibly off-rank local coordinate.
   * @param[in] axis Axis in x/y/z/t order.
   * @param[in] arg Global extents and rank offset.
   * @return Torus-wrapped global coordinate.
   */
  template <typename Arg> __device__ __host__ inline int global_coordinate(int value, int axis, const Arg &arg)
  {
    int coordinate = arg.global_offset[axis] + value;
    coordinate %= arg.global_X[axis];
    if (coordinate < 0) coordinate += arg.global_X[axis];
    return coordinate;
  }

  /**
   * Compute the spatial MILC link phase at a possibly off-rank coordinate.
   * The rotating level-2 X links already contain both the MILC phase and the
   * temporal fermion-boundary sign. This helper returns the x/y/z MILC sign
   * but returns +1 for t, so gauge_at strips only the MILC phase and preserves
   * the temporal boundary sign.
   *
   * StaggeredPhase in index_helper.cuh and getPhase in gauge_phase.cuh also
   * handle the temporal boundary and would remove the sign that must remain.
   * The unused milcStaggeredPhase reference in gauge_field_order.h evaluates
   * coordinates relative to a local halo and does not include the MPI rank's
   * global offset. Neither is a drop-in replacement for this helper.
   *
   * @param[in] axis Link direction in x/y/z/t order.
   * @param[in] c Local link-base coordinate.
   * @param[in] arg Rank-aware geometry.
   * @return Spatial MILC staggered phase baked into the link, +1 or -1.
   */
  template <typename Arg>
  __device__ __host__ inline int milc_sign(int axis, const int c[4], const Arg &arg)
  {
    int x = global_coordinate(c[0], 0, arg), y = global_coordinate(c[1], 1, arg);
    int t = global_coordinate(c[3], 3, arg);
    int s;
    switch (axis) {
    case 0: s = (t & 1); break;
    case 1: s = ((x + t) & 1); break;
    case 2: s = ((x + y + t) & 1); break;
    default: s = 0; break;
    }
    return s ? -1 : 1;
  }

  /** +axis fat link at (wrapped) full coord `c`, with the MILC staggered
   *  phase stripped (x: -1^t, y: -1^(x+t), z: -1^(x+y+t), t: +1).
   *  QUDA's resident HISQ links bake both the MILC phase and the fermion
   *  temporal-boundary sign. The rotating paths use bare links plus explicit
   *  eta factors, so remove only the MILC phase and leave the boundary sign
   *  on temporal hops.
   *  @param[in] axis Forward link direction in x/y/z/t order.
   *  @param[in] c Local link-base coordinate, possibly in the extended halo.
   *  @param[in] arg Extended gauge field and geometry.
   *  @return Link with its MILC phase removed.
   */
  template <typename Link, typename Arg>
  __device__ __host__ inline Link gauge_at(int axis, const int c[4], const Arg &arg)
  {
    int p = full_parity(c, arg.X);
    int extended[4];
    for (int d = 0; d < 4; d++) {
      extended[d] = arg.R[d] ? c[d] + arg.R[d] : wrap_ax(c[d], arg.X[d]);
    }
    int cb = linkIndex(extended, arg.E);
    Link u = arg.U(axis, cb, p);
    return static_cast<typename Arg::real>(milc_sign(axis, c, arg)) * u;
  }

  /**
   * @brief _deviceLinkT: ordered product of links along signed dir codes.
   *   code d>0: multiply U_axis(pos) then step +1 on axis.
   *   code d<0: step -1 on axis first, then multiply \f$U_axis(pos)^\dagger\f$.
   * dirs are {\f$\pm 1\f$:x, \f$\pm 2\f$:y, \f$\pm 3\f$:z, \f$\pm 4\f$:t}.
   * @param[in] start Path start in local full coordinates.
   * @param[in] dirs Signed one-based direction codes.
   * @param[in] len Number of hops in dirs.
   * @param[in] arg Extended gauge field and geometry.
   * @return Ordered link product from start to the endpoint.
   */
  template <typename Link, typename Arg>
  __device__ __host__ inline Link link_path(const int start[4], const int *dirs, int len, const Arg &arg)
  {
    Link ret;
    setIdentity(&ret);
    int pos[4] = {start[0], start[1], start[2], start[3]};
    for (int i = 0; i < len; i++) {
      int d = dirs[i];
      int axis = (d > 0 ? d : -d) - 1;
      if (d > 0) {
        ret = ret * gauge_at<Link>(axis, pos, arg);
        pos[axis] += 1;
      } else {
        pos[axis] -= 1;
        ret = ret * conj(gauge_at<Link>(axis, pos, arg));
      }
    }
    return ret;
  }
  /**
   * Return the symmetrized orbital transporter.
   * @param[in] n Path start in local full coordinates.
   * @param[in] bXorY Coordinate-weight axis: 0 for x and 1 for y.
   * @param[in] bPlusMu Sign of the repeated transverse hop.
   * @param[in] bPlusTau Sign of the temporal hop.
   * @param[in] arg Extended gauge field and geometry.
   * @return One third of the three orderings of {mu,mu,tau}.
   */
  template <typename Link, typename Arg>
  __device__ __host__ inline Link vxxtau(const int n[4], int bXorY, bool bPlusMu, bool bPlusTau, const Arg &arg)
  {
    int byMu = 1 - bXorY;
    int iMu = bPlusMu ? (byMu + 1) : -(byMu + 1);
    int iTau = bPlusTau ? 4 : -4;

    int d0[2] = {iMu, iTau};
    int d1[2] = {iTau, iMu};
    Link sRet = link_path<Link>(n, d0, 2, arg) + link_path<Link>(n, d1, 2, arg);

    int xt[4] = {n[0], n[1], n[2], n[3]};
    xt[3] += bPlusTau ? 1 : -1;
    xt[byMu] += bPlusMu ? 1 : -2;
    Link u = gauge_at<Link>(byMu, xt, arg);
    sRet = bPlusMu ? (sRet * u) : (sRet * conj(u));

    int d2[3] = {iMu, iMu, iTau};
    sRet = sRet + link_path<Link>(n, d2, 3, arg);
    return static_cast<typename Arg::real>(1.0 / 3.0) * sRet;
  }

  /**
   * Return the symmetrized spin transporter.
   * @param[in] n Path start in local full coordinates.
   * @param[in] bPlusX Sign of the x hop.
   * @param[in] bPlusY Sign of the y hop.
   * @param[in] bPlusT Sign of the temporal hop.
   * @param[in] arg Extended gauge field and geometry.
   * @return One sixth of all six orderings of {x,y,t}.
   */
  template <typename Link, typename Arg>
  __device__ __host__ inline Link vxyt(const int n[4], bool bPlusX, bool bPlusY, bool bPlusT, const Arg &arg)
  {
    int iX = bPlusX ? 1 : -1;
    int iY = bPlusY ? 2 : -2;
    int iT = bPlusT ? 4 : -4;

    int n_xy[4] = {n[0], n[1], n[2], n[3]};
    int n_xt[4] = {n[0], n[1], n[2], n[3]};
    int n_yt[4] = {n[0], n[1], n[2], n[3]};
    if (bPlusX) { n_xy[0] += 1; n_xt[0] += 1; } else { n_xy[0] -= 1; n_xt[0] -= 1; n_yt[0] -= 1; }
    if (bPlusY) { n_xy[1] += 1; n_yt[1] += 1; } else { n_xy[1] -= 1; n_yt[1] -= 1; n_xt[1] -= 1; }
    if (bPlusT) { n_xt[3] += 1; n_yt[3] += 1; } else { n_xt[3] -= 1; n_yt[3] -= 1; n_xy[3] -= 1; }

    int dxy[2] = {iX, iY}, dyx[2] = {iY, iX};
    Link s = link_path<Link>(n, dxy, 2, arg) + link_path<Link>(n, dyx, 2, arg);
    Link ut = gauge_at<Link>(3, n_xy, arg);
    s = bPlusT ? (s * ut) : (s * conj(ut));

    int dxt[2] = {iX, iT}, dtx[2] = {iT, iX};
    Link s2 = link_path<Link>(n, dxt, 2, arg) + link_path<Link>(n, dtx, 2, arg);
    Link uy = gauge_at<Link>(1, n_xt, arg);
    s2 = bPlusY ? (s2 * uy) : (s2 * conj(uy));
    s = s + s2;

    int dyt[2] = {iY, iT}, dty[2] = {iT, iY};
    s2 = link_path<Link>(n, dyt, 2, arg) + link_path<Link>(n, dty, 2, arg);
    Link ux = gauge_at<Link>(0, n_yt, arg);
    s2 = bPlusX ? (s2 * ux) : (s2 * conj(ux));
    s = s + s2;

    return static_cast<typename Arg::real>(1.0 / 6.0) * s;
  }

  template <typename Float, int nColor_, QudaReconstructType reconstruct_>
  struct HISQRotatingOrbitalSpinLinkCacheArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr QudaReconstructType reconstruct = reconstruct_;
    using G = typename gauge_mapper<Float, reconstruct>::type;

    G vxxtau_minus_t; /** Direction-packed VXXTau(-t) cache destination. */
    G vxxtau_plus_t;  /** Direction-packed VXXTau(+t) cache destination. */
    G vxyt_minus_t;   /** Direction-packed VXYT(-t) cache destination. */
    G vxyt_plus_t;    /** Direction-packed VXYT(+t) cache destination. */
    const G U;        /** Extended, phased pre-epsilon level-2 X source. */
    int E[4];         /** Extended X-link dimensions. */
    int R[4];         /** Extended X-link halo radii. */
    int global_X[4];      /** Global physical lattice dimensions. */
    int global_offset[4]; /** This rank's global physical origin. */
    int X[4];             /** Local physical cache dimensions. */

    /** @param[out] cache Direction-packed cache fields. @param[in] U Extended level-2 X links. */
    HISQRotatingOrbitalSpinLinkCacheArg(const MutableHISQRotatingOrbitalSpinLinkCache &cache, const GaugeField &U) :
      kernel_param(dim3(cache.vxxtau_minus_t->VolumeCB(), 2, 1)),
      vxxtau_minus_t(*cache.vxxtau_minus_t),
      vxxtau_plus_t(*cache.vxxtau_plus_t),
      vxyt_minus_t(*cache.vxyt_minus_t),
      vxyt_plus_t(*cache.vxyt_plus_t),
      U(U)
    {
      for (int i = 0; i < 4; i++) {
        X[i] = cache.vxxtau_minus_t->X()[i];
        E[i] = U.X()[i];
        R[i] = U.R()[i];
        global_X[i] = X[i] * comm_dim(i);
        global_offset[i] = X[i] * comm_coord(i);
      }
    }
  };

  template <typename Arg> struct HISQRotatingOrbitalSpinLinkCacheBuild {
    const Arg &arg;
    /** @param[in] arg Cache destinations and extended X source. */
    constexpr HISQRotatingOrbitalSpinLinkCacheBuild(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    /** @param[in] x_cb Checkerboard site. @param[in] parity Site parity. @param[in] z Unused kernel index. */
    __device__ __host__ inline void operator()(int x_cb, int parity, int z)
    {
      using Link = Matrix<complex<typename Arg::real>, Arg::nColor>;
      int n[4];
      getCoords(n, x_cb, arg.X, parity);

#pragma unroll
      for (int slot = 0; slot < 4; slot++) {
        int idx = slot;
        arg.vxxtau_minus_t(slot, x_cb, parity)
          = vxxtau<Link>(n, idx & 1, (idx & 2) != 0, (idx & 4) != 0, arg);
        idx = slot + 4;
        arg.vxxtau_plus_t(slot, x_cb, parity)
          = vxxtau<Link>(n, idx & 1, (idx & 2) != 0, (idx & 4) != 0, arg);

        idx = slot;
        arg.vxyt_minus_t(slot, x_cb, parity)
          = vxyt<Link>(n, (idx & 1) != 0, (idx & 2) != 0, (idx & 4) != 0, arg);
        idx = slot + 4;
        arg.vxyt_plus_t(slot, x_cb, parity)
          = vxyt<Link>(n, (idx & 1) != 0, (idx & 2) != 0, (idx & 4) != 0, arg);
      }
    }
  };

  /**
   * @param[in] arg Cached or on-demand path source.
   * @param[in] n Path start coordinate.
   * @param[in] idx Packed coordinate axis and hop signs.
   * @param[in] x_cb Cache checkerboard site.
   * @param[in] parity Cache site parity.
   * @return Requested VXXTau path matrix.
   */
  template <typename Link, typename Arg>
  __device__ __host__ inline Link load_vxxtau(const Arg &arg, const int n[4], int idx, int x_cb, int parity)
  {
    if constexpr (Arg::cache_rotation_links) {
      return idx < 4 ? arg.vxxtau_minus_t(idx, x_cb, parity)
                     : arg.vxxtau_plus_t(idx - 4, x_cb, parity);
    } else {
      return vxxtau<Link>(n, idx & 1, (idx & 2) != 0, (idx & 4) != 0, arg);
    }
  }

  /**
   * @param[in] arg Cached or on-demand path source.
   * @param[in] n Path start coordinate.
   * @param[in] idx Packed x/y/t hop signs.
   * @param[in] x_cb Cache checkerboard site.
   * @param[in] parity Cache site parity.
   * @return Requested VXYT path matrix.
   */
  template <typename Link, typename Arg>
  __device__ __host__ inline Link load_vxyt(const Arg &arg, const int n[4], int idx, int x_cb, int parity)
  {
    if constexpr (Arg::cache_rotation_links) {
      return idx < 4 ? arg.vxyt_minus_t(idx, x_cb, parity)
                     : arg.vxyt_plus_t(idx - 4, x_cb, parity);
    } else {
      return vxyt<Link>(n, (idx & 1) != 0, (idx & 2) != 0, (idx & 4) != 0, arg);
    }
  }

  /**
   * @brief D_rot correction at site n: XY orbital term (\f$0.25\Omega\f$) and XYTau term
   * (\f$0.125\Omega\f$).
   *
   * On the supported torus, the orbital phase reduces to eta4 and the
   * polarization phase to eta124 at the current site. The temporal
   * anti-periodic sign is already carried by every temporal fat-link hop.
   * Adds onto out[].
   * @param[in,out] out Register tile containing the ordinary Dslash result.
   * @param[in] arg Spinors, links, cache, halo, and rotation parameters.
   * @param[in] x_cb Output checkerboard site.
   * @param[in] parity Output parity.
   * @param[in] src_idx First right-hand-side index represented by out.
   */
  template <int n_src_tile, typename Arg, typename Vector>
  __device__ __host__ inline void applyStaggeredRotating(array<Vector, n_src_tile> &out, const Arg &arg, int x_cb,
                                                          int parity, int src_idx)
  {
    using real = typename Arg::real;
    using Link = Matrix<complex<real>, Arg::nColor>;

    int n[4];
    getCoords(n, x_cb, arg.X, parity);
    const real shift = static_cast<real>(0.5); // shift-center

    auto psi_at = [&](const int tgt[4], int s) -> Vector {
      if constexpr (Arg::use_spinor_halo) {
        int wrapped[4];
        if (rotating_remote_site(tgt, wrapped, arg)) {
          const int rhs = src_idx + s;
          const size_t halo_site = rotating_halo_index(tgt, arg);
          const size_t base = (halo_site * arg.halo_rhs_capacity + rhs) * Arg::nColor;
          Vector vector;
          for (int color = 0; color < Arg::nColor; color++) vector(color) = arg.halo[base + color];
          return vector;
        } else {
          const int p = full_parity(wrapped, arg.X);
          const int cb = full_cb_index(wrapped, arg.X);
          const int spinor_parity = arg.nParity == 2 ? p : 0;
          return arg.in[src_idx + s](cb, spinor_parity);
        }
      } else {
        int p = full_parity(tgt, arg.X);
        int cb = full_cb_index(tgt, arg.X);
        int spinor_parity = arg.nParity == 2 ? p : 0;
        return arg.in[src_idx + s](cb, spinor_parity);
      }
    };

    const real xy_coeff = static_cast<real>(0.25) * arg.angular_velocity * arg.dagger_scale;
    const real tau_coeff = static_cast<real>(0.125) * arg.angular_velocity * arg.dagger_scale;

    // ---- XY orbital term ----
#pragma unroll
    for (int idx = 0; idx < 8; idx++) {
      int bXorY = idx & 1;
      bool bPlusMu = (idx & 2) != 0;
      bool bPlusTau = (idx & 4) != 0;
      int bYorX = 1 - bXorY;

      int target[4] = {n[0], n[1], n[2], n[3]};
      target[bYorX] += bPlusMu ? 2 : -2;
      target[3] += bPlusTau ? 1 : -1;

      int mid[4] = {n[0], n[1], n[2], n[3]};
      mid[bYorX] += bPlusMu ? 1 : -1;
      int midw = global_coordinate(mid[bXorY], bXorY, arg);

      int phase = bYorX; // eta_tau == 0 (MILC)
      if (!bPlusMu) phase += 1;

      real weight = static_cast<real>(midw - arg.center[bXorY]) + shift;
      real sgn = (phase & 1) ? static_cast<real>(-1.0) : static_cast<real>(1.0);
      real coeff = xy_coeff * weight * sgn;

      Link V = load_vxxtau<Link>(arg, n, idx, x_cb, parity);
#pragma unroll
      for (int s = 0; s < n_src_tile; s++) out[s] = out[s] + coeff * (V * psi_at(target, s));
    }

    // ---- XYTau polarization term ----
#pragma unroll
    for (int idx = 0; idx < 8; idx++) {
      bool bPlusX = (idx & 1) != 0;
      bool bPlusY = (idx & 2) != 0;
      bool bPlusT = (idx & 4) != 0;

      int target[4] = {n[0], n[1], n[2], n[3]};
      target[0] += bPlusX ? 1 : -1;
      target[1] += bPlusY ? 1 : -1;
      target[3] += bPlusT ? 1 : -1;

      int e;
      if (bPlusT) {
        e = global_coordinate(n[0], 0, arg) & 1;
      } else {
        e = (global_coordinate(target[0], 0, arg) & 1) + 1;
      }

      real sgn = (e & 1) ? static_cast<real>(-1.0) : static_cast<real>(1.0);
      real coeff = tau_coeff * sgn;

      Link V = load_vxyt<Link>(arg, n, idx, x_cb, parity);
#pragma unroll
      for (int s = 0; s < n_src_tile; s++) out[s] = out[s] + coeff * (V * psi_at(target, s));
    }
  }

  template <typename Arg> struct StaggeredRotatingApply {
    const Arg &arg;
    /** @param[in] arg Dslash fields and rotating parameters. */
    constexpr StaggeredRotatingApply(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    /** @param[in] x_cb Output site. @param[in] src_idx Right-hand side. @param[in] z Unused index. */
    __device__ __host__ inline void operator()(int x_cb, int src_idx, int z)
    {
      using Vector = ColorSpinor<typename Arg::real, Arg::nColor, Arg::nSpin>;
      array<Vector, 1> out;
      const int spinor_parity = arg.nParity == 2 ? arg.parity : 0;
      out[0] = arg.out[src_idx](x_cb, spinor_parity); // additive correction onto base D
      applyStaggeredRotating<1>(out, arg, x_cb, arg.parity, src_idx);
      arg.out[src_idx](x_cb, spinor_parity) = out[0];
    }
  };

} // namespace quda
