#pragma once

#include <array>
#include <cstddef>

#include <color_spinor_field.h>
#include <quda_ptr.h>

namespace quda
{
  /**
     Compact corner-halo storage for the rotating staggered stencil.

     The rotating endpoints cross x/y/t simultaneously, so ordinary face-only
     spinor ghosts are insufficient.  This class packs the disjoint remote
     coordinate regions for all neighboring ranks directly into raw color
     vectors.  It avoids the full physical-volume GaugeField formerly used
     only to borrow extended-gauge corner exchange.  Send and receive device
     buffers retain the largest right-hand-side batch seen by this object;
     non-GDR communication additionally uses equally sized pinned host
     staging buffers.
   */
  class StaggeredRotatingHalo
  {
  public:
    static constexpr int region_count = 27; // {-1,0,+1} in x/y/t

  private:
    quda_ptr send_device; /** Owned packed device send buffer. */
    quda_ptr recv_device; /** Owned packed device receive buffer. */
    quda_ptr send_host;   /** Owned pinned send staging buffer when GDR is unavailable. */
    quda_ptr recv_host;   /** Owned pinned receive staging buffer when GDR is unavailable. */
    std::array<size_t, region_count> region_offset = {}; /** Site offset for each {-1,0,+1} x/y/t region. */
    std::array<size_t, region_count> region_sites = {};  /** Packed site count for each neighbor region. */
    std::array<int, 4> x = {}; /** Local physical spinor extents in x/y/z/t order. */
    std::array<int, 4> r = {}; /** Maximum rotating endpoint displacement in each direction. */
    size_t sites = 0;          /** Total packed sites over all active remote regions. */
    size_t bytes = 0;          /** Bytes in one send or receive allocation. */
    int rhs_capacity = 0;      /** Largest right-hand-side batch retained by the buffers. */
    QudaPrecision precision = QUDA_INVALID_PRECISION; /** Real storage precision of packed color vectors. */

    /**
       Resize and describe buffers for the requested spinor batch.
       @param[in] in Representative input spinor defining local geometry and precision.
       @param[in] n_rhs Number of right-hand sides in the upcoming exchange.
     */
    void configure(const ColorSpinorField &in, int n_rhs);

  public:
    StaggeredRotatingHalo() = default;
    StaggeredRotatingHalo(const StaggeredRotatingHalo &) = delete;
    StaggeredRotatingHalo &operator=(const StaggeredRotatingHalo &) = delete;

    /**
       Pack and exchange all remote x/y/t face, edge, and corner regions.
       @param[in] in Input right-hand-side batch.
       @param[in] output_parity Requested Dslash output parity, or QUDA_INVALID_PARITY.
       @return This workspace with receive data ready for device access.
     */
    const StaggeredRotatingHalo &exchange(cvector_ref<const ColorSpinorField> &in, int output_parity);
    /** Release all retained device and host buffers and reset geometry. */
    void reset();

    /** @return Receive-buffer pointer, or nullptr for a local-only stencil. */
    const void *data() const { return bytes ? recv_device.data_device() : nullptr; }
    /** @return Send-buffer pointer, or nullptr for a local-only stencil. */
    void *sendData() const { return bytes ? send_device.data_device() : nullptr; }
    /** @return Site offset of each encoded neighbor region. */
    const std::array<size_t, region_count> &RegionOffset() const { return region_offset; }
    /** @return Number of sites in each encoded neighbor region. */
    const std::array<size_t, region_count> &RegionSites() const { return region_sites; }
    /** @return Local physical spinor extents. */
    const std::array<int, 4> &X() const { return x; }
    /** @return Extended X-link halo radii. */
    const std::array<int, 4> &R() const { return r; }
    /** @return Total packed sites over active remote regions. */
    size_t Sites() const { return sites; }
    /** @return Bytes used by one packed send or receive buffer. */
    size_t Bytes() const { return bytes; }
    /** @return Largest right-hand-side batch retained by this workspace. */
    int RHSCapacity() const { return rhs_capacity; }
  };
} // namespace quda
