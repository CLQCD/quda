#pragma once

#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <index_helper.cuh>
#include <kernel.h>
#include <kernels/dslash_staggered_rotating.cuh>

// Rotating-frame contribution to the HISQ one-link derivative field f0.
// Paths are evaluated on bare fat links: gauge_at strips QUDA's baked MILC
// phase but intentionally retains the temporal boundary sign.  The resulting
// matrix is converted to QUDA's staple-oprod convention before the ordinary
// HISQ smearing backward pass.

namespace quda
{

  template <typename Float, int nColor_> struct HisqForceRotatingArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false;
    using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
    using F0 = gauge::FieldOrder<real, nColor, 1, QUDA_NATIVE_GAUGE_ORDER>;
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, true>::type;

    F0 f0;         /** Non-owning level-2 force accumulator in direction-major native layout. */
    const G U;     /** Extended, phased pre-epsilon level-2 X links. */
    const complex<real> *halo; /** Non-owning compact remote-spinor buffer; null for a local stencil. */
    const F in;    /** Full even/odd rational-solution field for locally owned endpoints. */
    const real angular_velocity; /** Rotation angular velocity in lattice units. */
    const real coeff;            /** Rational-residue force coefficient. */

    int X[4];           /** Local physical spinor dimensions in x/y/z/t order. */
    int E[4];           /** Extended X-link dimensions in x/y/z/t order. */
    int R[4];           /** Extended X-link halo radius on each side. */
    int partitioned[4]; /** Whether the communicator is partitioned in each direction. */
    size_t halo_offset[StaggeredRotatingHalo::region_count]; /** Packed receive offset for each x/y/t region. */
    int halo_rhs_capacity; /** Right-hand-side stride of the packed halo allocation. */
    int global_X[4];       /** Global physical dimensions in x/y/z/t order. */
    int global_offset[4];  /** This rank's global physical origin. */
    int center[2];         /** Derived global x/y midpoint for shift-center-half coordinates. */
    int volume_cb;         /** Half of the local physical four-volume. */

    /**
       @param[in,out] f0 Level-2 force accumulator.
       @param[in] U Extended level-2 X links.
       @param[in] halo Packed remote endpoint spinors.
       @param[in] in Full even/odd rational-solution field.
       @param[in] angular_velocity Rotation angular velocity in lattice units.
       @param[in] coeff Rational-residue force coefficient.
     */
    HisqForceRotatingArg(GaugeField &f0, const GaugeField &U, const StaggeredRotatingHalo &halo,
                         const ColorSpinorField &in, double angular_velocity, double coeff) :
      kernel_param(dim3(in.VolumeCB() * 8, 1, 1)),
      f0(f0),
      U(U),
      halo(static_cast<const complex<real> *>(halo.data())),
      in(in),
      angular_velocity(angular_velocity),
      coeff(coeff),
      halo_rhs_capacity(halo.RHSCapacity()),
      volume_cb(in.VolumeCB())
    {
      for (int d = 0; d < 4; d++) {
        X[d] = in.X()[d];
        E[d] = U.X()[d];
        R[d] = U.R()[d];
        partitioned[d] = comm_dim_partitioned(d);
        global_X[d] = X[d] * comm_dim(d);
        global_offset[d] = X[d] * comm_coord(d);
      }
      center[0] = global_X[0] / 2;
      center[1] = global_X[1] / 2;
      for (int region = 0; region < StaggeredRotatingHalo::region_count; region++)
        halo_offset[region] = halo.RegionOffset()[region];
    }
  };

  /**
     @param[in] arg Local spinor field and compact halo.
     @param[in] site Possibly off-rank endpoint coordinates.
     @return Color vector at site.
   */
  template <typename Arg, typename Vector>
  __device__ __host__ inline Vector rotating_spinor_at(const Arg &arg, const int site[4])
  {
    int wrapped[4];
    if (rotating_remote_site(site, wrapped, arg)) {
      const size_t halo_site = rotating_halo_index(site, arg);
      const size_t base = halo_site * arg.halo_rhs_capacity * Arg::nColor;
      Vector vector;
      for (int color = 0; color < Arg::nColor; color++) vector(color) = arg.halo[base + color];
      return vector;
    } else {
      const int parity = full_parity(wrapped, arg.X);
      return arg.in(full_cb_index(wrapped, arg.X), parity);
    }
  }

  /**
   * Recover the path start whose step k traverses the owned forward link.
   * For a backward step the path position is one site beyond the link base.
   * @param[in] owned Base coordinate of the differentiated forward link.
   * @param[in] dirs Three signed one-based path directions.
   * @param[in] k Position of the differentiated occurrence in dirs.
   * @param[out] start Recovered path-start coordinate.
   */
  template <typename Arg>
  __device__ __host__ inline void rotating_path_start(const int owned[4], const int dirs[3], int k, int start[4])
  {
    for (int d = 0; d < 4; d++) start[d] = owned[d];
    for (int q = 0; q < k; q++) {
      const int direction = dirs[q];
      const int axis = (direction > 0 ? direction : -direction) - 1;
      start[axis] -= direction > 0 ? 1 : -1;
    }
    if (dirs[k] < 0) {
      const int axis = -dirs[k] - 1;
      start[axis] += 1;
    }
  }

  /**
   * Add the derivative of one path occurrence to a register accumulator.
   * The differentiated link is excluded from both path products.  This is the
   * same left/right separation used by CLGLib's link-owned force kernels.
   * @param[in] arg Gauge, spinor, halo, and coordinate metadata.
   * @param[in] owned Base coordinate of the differentiated forward link.
   * @param[in] parity Parity of owned.
   * @param[in] axis Direction of the differentiated forward link.
   * @param[in] start Path-start coordinate.
   * @param[in] dirs Three signed one-based path directions.
   * @param[in] k Position of the differentiated occurrence.
   * @param[in] c Path coefficient before MILC/deposit convention factors.
   * @param[in,out] sum Register matrix accumulating this link's force.
   */
  template <typename Arg, typename Link, typename Vector>
  __device__ __host__ inline void accumulate_rotating_occurrence(const Arg &arg, const int owned[4], int parity,
                                                                  int axis, const int start[4], const int dirs[3],
                                                                  int k, typename Arg::real c, Link &sum)
  {
    using real = typename Arg::real;
    Link factors[3];
    int position[4] = {start[0], start[1], start[2], start[3]};
    for (int q = 0; q < 3; q++) {
      const int direction = dirs[q];
      const int direction_axis = (direction > 0 ? direction : -direction) - 1;
      if (direction > 0) {
        if (q != k) factors[q] = gauge_at<Link>(direction_axis, position, arg);
        position[direction_axis] += 1;
      } else {
        position[direction_axis] -= 1;
        if (q != k) factors[q] = conj(gauge_at<Link>(direction_axis, position, arg));
      }
    }

    const Vector a = rotating_spinor_at<Arg, Vector>(arg, start);
    const Vector b = rotating_spinor_at<Arg, Vector>(arg, position);
    Vector left = a;
    for (int q = 0; q < k; q++) left = conj(factors[q]) * left;
    Vector right = b;
    for (int q = 2; q > k; q--) right = factors[q] * right;

    real cc = c * static_cast<real>(milc_sign(axis, owned, arg));
    cc *= parity ? static_cast<real>(-1.0) : static_cast<real>(1.0);
    cc = -cc;

    sum += dirs[k] > 0 ? cc * outerProduct(right, left) : cc * outerProduct(left, right);
  }

  /**
   * One thread owns one local forward link.  The one-dimensional thread index
   * is a bijection with (axis, parity, x_cb): there are exactly
   * 4 * 2 * volume_cb threads and no two indices decode to the same tuple.
   * Each thread enumerates every rotating three-link path containing its
   * owned link, accumulates only in a register matrix, then performs the only
   * f0 write in this kernel once, at the owned tuple.  This is the fused
   * equivalent of CLGLib's serial path/separation launches, in each of which
   * uiSiteIndex owns _deviceGetLinkIndex(uiSiteIndex, dir).  It is therefore
   * race-free without atomics.  Extended gauge and packed-spinor halos provide
   * off-rank endpoints and path factors; the output link always remains local.
   */
  template <typename Arg> struct HisqForceRotatingLinkOprod {
    const Arg &arg;
    /** @param[in] arg Link-owned force fields and coefficients. */
    constexpr HisqForceRotatingLinkOprod(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    /** @param[in] link_index Bijection over local (axis, parity, checkerboard site). */
    __device__ __host__ inline void operator()(int link_index)
    {
      using real = typename Arg::real;
      using Link = Matrix<complex<real>, Arg::nColor>;
      using Vector = ColorSpinor<real, Arg::nColor, Arg::nSpin>;

      const int x_cb = link_index % arg.volume_cb;
      const int parity = (link_index / arg.volume_cb) % 2;
      const int axis = link_index / (2 * arg.volume_cb);
      int owned[4];
      getCoords(owned, x_cb, arg.X, parity);
      Link sum;
#pragma unroll
      for (int row = 0; row < Arg::nColor; row++)
#pragma unroll
        for (int column = 0; column < Arg::nColor; column++) sum(row, column) = 0.0;

      // Orbital paths: three orderings of {mu,mu,t} for all hop signs.
      for (int bXorY = 0; bXorY < 2; bXorY++) {
        const int byMu = 1 - bXorY;
        for (int sign_bits = 0; sign_bits < 4; sign_bits++) {
          const bool plus_mu = (sign_bits & 1) != 0;
          const bool plus_t = (sign_bits & 2) != 0;
          const int mu = plus_mu ? byMu + 1 : -(byMu + 1);
          const int tau = plus_t ? 4 : -4;
          const int paths[3][3] = {{mu, mu, tau}, {mu, tau, mu}, {tau, mu, mu}};

          for (int order = 0; order < 3; order++) {
            for (int k = 0; k < 3; k++) {
              const int direction = paths[order][k];
              if ((direction > 0 ? direction : -direction) - 1 != axis) continue;

              int start[4];
              rotating_path_start<Arg>(owned, paths[order], k, start);
              if (full_parity(start, arg.X) != 1) continue;

              int middle[4] = {start[0], start[1], start[2], start[3]};
              middle[byMu] += plus_mu ? 1 : -1;

              const real weight = static_cast<real>(global_coordinate(middle[bXorY], bXorY, arg) - arg.center[bXorY])
                + static_cast<real>(0.5);
              int phase = byMu;
              if (!plus_mu) phase++;
              const real sign = phase & 1 ? static_cast<real>(-1.0) : static_cast<real>(1.0);
              const real c = arg.coeff * static_cast<real>(0.25 / 3.0) * arg.angular_velocity * sign * weight;
              if (c != static_cast<real>(0.0))
                accumulate_rotating_occurrence<Arg, Link, Vector>(arg, owned, parity, axis, start,
                                                                  paths[order], k, c, sum);
            }
          }
        }
      }

      // Polarization paths: six orderings of {x,y,t} for all hop signs.
      for (int sign_bits = 0; sign_bits < 8; sign_bits++) {
        const bool plus_x = (sign_bits & 1) != 0;
        const bool plus_y = (sign_bits & 2) != 0;
        const bool plus_t = (sign_bits & 4) != 0;
        const int x_dir = plus_x ? 1 : -1;
        const int y_dir = plus_y ? 2 : -2;
        const int t_dir = plus_t ? 4 : -4;
        const int paths[6][3] = {{x_dir, y_dir, t_dir}, {x_dir, t_dir, y_dir},
                                 {y_dir, x_dir, t_dir}, {y_dir, t_dir, x_dir},
                                 {t_dir, x_dir, y_dir}, {t_dir, y_dir, x_dir}};

        for (int order = 0; order < 6; order++) {
          for (int k = 0; k < 3; k++) {
            const int direction = paths[order][k];
            if ((direction > 0 ? direction : -direction) - 1 != axis) continue;

            int start[4];
            rotating_path_start<Arg>(owned, paths[order], k, start);
            if (full_parity(start, arg.X) != 1) continue;

            const int target_x = start[0] + (plus_x ? 1 : -1);
            int eta = plus_t ? global_coordinate(start[0], 0, arg) & 1
                             : (global_coordinate(target_x, 0, arg) & 1) + 1;
            const real sign = eta & 1 ? static_cast<real>(-1.0) : static_cast<real>(1.0);
            const real c = arg.coeff * static_cast<real>(0.125 / 6.0) * arg.angular_velocity * sign;
            accumulate_rotating_occurrence<Arg, Link, Vector>(arg, owned, parity, axis, start,
                                                              paths[order], k, c, sum);
          }
        }
      }

#pragma unroll
      for (int row = 0; row < Arg::nColor; row++)
#pragma unroll
        for (int column = 0; column < Arg::nColor; column++) {
          const complex<real> previous = arg.f0(axis, parity, x_cb, 0, 0, row, column);
          arg.f0(axis, parity, x_cb, 0, 0, row, column) = previous + sum(row, column);
        }
    }
  };

} // namespace quda
