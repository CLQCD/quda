#pragma once

#include <color_spinor_field_order.h>
#include <color_spinor.h>
#include <index_helper.cuh>
#include <kernel.h>
#include <staggered_rotating_halo.h>

namespace quda
{
  /** @param[in] value Coordinate to wrap. @param[in] extent Torus extent. @return Wrapped coordinate. */
  __device__ __host__ inline int rotating_halo_wrap(int value, int extent)
  {
    return ((value % extent) + extent) % extent;
  }

  /** @param[in] region Encoded x/y/t neighbor region. @param[out] displacement Components in {-1,0,+1}. */
  __device__ __host__ inline void rotating_halo_displacement(int region, int displacement[4])
  {
    displacement[3] = region % 3 - 1;
    region /= 3;
    displacement[1] = region % 3 - 1;
    region /= 3;
    displacement[0] = region % 3 - 1;
    displacement[2] = 0;
  }

  /**
     @param[in] site Possibly off-rank endpoint coordinates.
     @param[in] arg Halo offsets, radii, extents, and partitioning metadata.
     @return Linear site index in the compact receive halo.
   */
  template <typename Arg>
  __device__ __host__ inline size_t rotating_halo_index(const int site[4], const Arg &arg)
  {
    int displacement[4] = {};
    for (int d = 0; d < 4; d++) {
      if (d == 2) continue;
      if (!arg.partitioned[d]) continue;
      displacement[d] = site[d] < 0 ? -1 : (site[d] >= arg.X[d] ? 1 : 0);
    }
    const int region = ((displacement[0] + 1) * 3 + displacement[1] + 1) * 3 + displacement[3] + 1;
    const size_t offset = arg.halo_offset[region];

    int coordinate[4];
    int extent[4];
    for (int d = 0; d < 4; d++) {
      extent[d] = displacement[d] == 0 ? arg.X[d] : arg.R[d];
      if (displacement[d] < 0) {
        coordinate[d] = site[d] + arg.R[d];
      } else if (displacement[d] > 0) {
        coordinate[d] = site[d] - arg.X[d];
      } else {
        coordinate[d] = rotating_halo_wrap(site[d], arg.X[d]);
      }
    }
    return offset + (((size_t)coordinate[3] * extent[2] + coordinate[2]) * extent[1] + coordinate[1])
      * extent[0] + coordinate[0];
  }

  template <typename Float, int nColor_> struct StaggeredRotatingHaloPackArg : kernel_param<> {
    using real = typename mapper<Float>::type;
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 1;
    static constexpr bool spin_project = false;
    static constexpr bool spinor_direct_load = false;
    using F = typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, true>::type;

    F in[MAX_MULTI_RHS]; /** Non-owning local input-spinor accessors. */
    complex<real> *send; /** Non-owning component-interleaved packed device destination. */
    size_t region_offset[StaggeredRotatingHalo::region_count]; /** Site offset of each x/y/t region. */
    size_t region_sites[StaggeredRotatingHalo::region_count];  /** Site count of each x/y/t region. */
    int X[4];          /** Local physical spinor extents. */
    int R[4];          /** Rotating endpoint radius in each direction. */
    int n_rhs;         /** Active right-hand-side count in this exchange. */
    int rhs_capacity;  /** Allocated right-hand-side stride in the packed buffer. */
    int n_parity;      /** Number of checkerboards stored by each input spinor. */
    int output_parity; /** Requested output parity, or QUDA_INVALID_PARITY for a full field. */

    /**
       @param[in] in Input spinor right-hand sides.
       @param[in,out] halo Send buffer and region layout.
       @param[in] output_parity_ Dslash output parity, or QUDA_INVALID_PARITY for a full field.
     */
    StaggeredRotatingHaloPackArg(cvector_ref<const ColorSpinorField> &in, const StaggeredRotatingHalo &halo,
                                 int output_parity_) :
      kernel_param(dim3(halo.Sites(), in.size(), 1)),
      send(static_cast<complex<real> *>(halo.sendData())),
      n_rhs(in.size()),
      rhs_capacity(halo.RHSCapacity()),
      n_parity(in[0].SiteSubset()),
      output_parity(output_parity_)
    {
      for (int d = 0; d < 4; d++) {
        X[d] = halo.X()[d];
        R[d] = halo.R()[d];
      }
      for (int region = 0; region < StaggeredRotatingHalo::region_count; region++) {
        region_offset[region] = halo.RegionOffset()[region];
        region_sites[region] = halo.RegionSites()[region];
      }
      for (auto i = 0u; i < in.size(); i++) this->in[i] = in[i];
    }

  };

  template <typename Arg> struct StaggeredRotatingHaloPack {
    const Arg &arg;
    /** @param[in] arg Input spinors and packed-halo layout. */
    constexpr StaggeredRotatingHaloPack(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    /** @param[in] halo_index Compact send-halo site. @param[in] rhs Right-hand-side index. */
    __device__ __host__ void operator()(int halo_index, int rhs)
    {
      using Vector = ColorSpinor<typename Arg::real, Arg::nColor, 1>;

      int region = 0;
      while (region < StaggeredRotatingHalo::region_count
             && (arg.region_sites[region] == 0
                 || static_cast<size_t>(halo_index) >= arg.region_offset[region] + arg.region_sites[region]))
        region++;
      if (region == StaggeredRotatingHalo::region_count) return;

      int displacement[4];
      rotating_halo_displacement(region, displacement);
      size_t local = static_cast<size_t>(halo_index) - arg.region_offset[region];
      int extent[4];
      for (int d = 0; d < 4; d++) extent[d] = displacement[d] == 0 ? arg.X[d] : arg.R[d];

      int coordinate[4];
      coordinate[0] = local % extent[0];
      local /= extent[0];
      coordinate[1] = local % extent[1];
      local /= extent[1];
      coordinate[2] = local % extent[2];
      local /= extent[2];
      coordinate[3] = local;
      for (int d = 0; d < 4; d++) {
        if (displacement[d] < 0) coordinate[d] += arg.X[d] - arg.R[d];
      }

      const int physical_parity
        = (coordinate[0] + coordinate[1] + coordinate[2] + coordinate[3]) & 1;
      const bool available = arg.n_parity == 2 || physical_parity == (arg.output_parity ^ 1);
      Vector vector;
      if (available) {
        const int x_cb = (((coordinate[3] * arg.X[2] + coordinate[2]) * arg.X[1] + coordinate[1])
                          * arg.X[0] + coordinate[0]) >> 1;
        vector = arg.in[rhs](x_cb, arg.n_parity == 2 ? physical_parity : 0);
      }

      const size_t base = (static_cast<size_t>(halo_index) * arg.rhs_capacity + rhs) * Arg::nColor;
      for (int color = 0; color < Arg::nColor; color++)
        arg.send[base + color] = available ? vector(color) : complex<typename Arg::real>(0.0, 0.0);
    }
  };
} // namespace quda
