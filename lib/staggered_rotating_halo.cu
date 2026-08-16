#include <array>
#include <limits>

#include <comm_quda.h>
#include <instantiate.h>
#include <kernels/staggered_rotating_halo.cuh>
#include <staggered_rotating_halo.h>
#include <tunable_nd.h>
#include <util_quda.h>

namespace quda
{
  namespace
  {
    template <typename Float, int nColor> class StaggeredRotatingHaloPackApply : public TunableKernel2D
    {
      cvector_ref<const ColorSpinorField> &in; /** Non-owning input right-hand-side batch. */
      StaggeredRotatingHalo &halo;             /** Owned workspace receiving packed send data. */
      int output_parity;                       /** Output parity, or QUDA_INVALID_PARITY for a full field. */

      unsigned int minThreads() const override { return halo.Sites(); }

    public:
      /**
         @param[in] in Input spinor right-hand sides.
         @param[in,out] halo Packed send buffer and region layout.
         @param[in] output_parity Dslash output parity, or QUDA_INVALID_PARITY.
       */
      StaggeredRotatingHaloPackApply(cvector_ref<const ColorSpinorField> &in, StaggeredRotatingHalo &halo,
                                     int output_parity) :
        TunableKernel2D(halo.Sites(), in.size(), QUDA_CUDA_FIELD_LOCATION),
        in(in), halo(halo), output_parity(output_parity)
      {
        strcat(aux, ",compact-rotating-halo,rhs=");
        u32toa(aux + strlen(aux), in.size());
        strcat(aux, comm_dim_partitioned_string());
        apply(device::get_default_stream());
      }

      /** @param[in] stream Device stream used for halo packing. */
      void apply(const qudaStream_t &stream) override
      {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        StaggeredRotatingHaloPackArg<Float, nColor> arg(in, halo, output_parity);
        launch<StaggeredRotatingHaloPack>(tp, stream, arg);
      }

      long long bytes() const override
      {
        return 2ll * halo.Sites() * in.size() * in[0].Ncolor() * 2 * in[0].Precision();
      }
    };

    /** @param[in] region Encoded x/y/t neighbor region. @param[out] displacement Neighbor displacement. */
    void decode_region(int region, int displacement[4])
    {
      displacement[3] = region % 3 - 1;
      region /= 3;
      displacement[1] = region % 3 - 1;
      region /= 3;
      displacement[0] = region % 3 - 1;
      displacement[2] = 0;
    }
  } // namespace

  void StaggeredRotatingHalo::configure(const ColorSpinorField &in, int n_rhs)
  {
    std::array<int, 4> required_x;
    std::array<int, 4> required_r;
    for (int d = 0; d < 4; d++) {
      required_x[d] = in.X()[d];
      if (in.SiteSubset() == QUDA_PARITY_SITE_SUBSET && d == 0) required_x[d] *= 2;
      const int stencil_radius[4] = {2, 2, 0, 1};
      required_r[d] = comm_dim_partitioned(d) ? stencil_radius[d] : 0;
    }

    const bool geometry_changed = x != required_x || r != required_r || precision != in.Precision();
    if (geometry_changed) {
      reset();
      x = required_x;
      r = required_r;
      precision = in.Precision();
      region_offset.fill(std::numeric_limits<size_t>::max());
      region_sites.fill(0);
      sites = 0;
      for (int region = 0; region < region_count; region++) {
        int displacement[4];
        decode_region(region, displacement);
        bool active = false;
        bool valid = true;
        size_t volume = 1;
        for (int d = 0; d < 4; d++) {
          if (displacement[d] != 0) {
            active = true;
            valid &= r[d] > 0;
            volume *= r[d];
          } else {
            volume *= x[d];
          }
        }
        if (!active || !valid) continue;
        region_offset[region] = sites;
        region_sites[region] = volume;
        sites += volume;
      }
    }

    if (n_rhs <= rhs_capacity) return;
    rhs_capacity = n_rhs;
    bytes = sites * rhs_capacity * in.Ncolor() * 2 * static_cast<size_t>(precision);
    send_device = quda_ptr(QUDA_MEMORY_DEVICE, bytes);
    recv_device = quda_ptr(QUDA_MEMORY_DEVICE, bytes);
    send_host = quda_ptr(QUDA_MEMORY_HOST_PINNED, bytes, false);
    recv_host = quda_ptr(QUDA_MEMORY_HOST_PINNED, bytes, false);
  }

  const StaggeredRotatingHalo &StaggeredRotatingHalo::exchange(cvector_ref<const ColorSpinorField> &in,
                                                               int output_parity)
  {
    if (in.empty() || in.size() > MAX_MULTI_RHS)
      errorQuda("Rotating halo batch size %lu is outside [1,%d]", in.size(), MAX_MULTI_RHS);
    if (in[0].Location() != QUDA_CUDA_FIELD_LOCATION)
      errorQuda("Rotating spinor halo requires a device input field");
    for (auto i = 1u; i < in.size(); i++) {
      checkPrecision(in[0], in[i]);
      checkLocation(in[0], in[i]);
      if (in[i].SiteSubset() != in[0].SiteSubset()) errorQuda("Rotating halo site-subset mismatch");
    }
    if (in[0].Ncolor() != 3) errorQuda("Rotating spinor halo only supports Ncolor=3");
    if (in[0].SiteSubset() == QUDA_PARITY_SITE_SUBSET && output_parity == QUDA_INVALID_PARITY)
      errorQuda("Rotating parity halo requires a valid output parity");

    configure(in[0], in.size());
    if (sites == 0) errorQuda("Rotating spinor halo exchange requested without a partitioned x/y/t direction");

    if (in[0].Precision() == QUDA_DOUBLE_PRECISION) {
      if constexpr (is_enabled(QUDA_DOUBLE_PRECISION))
        StaggeredRotatingHaloPackApply<double, 3>(in, *this, output_parity);
      else
        errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
    } else if (in[0].Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
        StaggeredRotatingHaloPackApply<float, 3>(in, *this, output_parity);
      else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
    } else {
      errorQuda("Rotating spinor halo only supports double and single precision, got %d", in[0].Precision());
    }
    qudaDeviceSynchronize();

    const bool gdr = comm_gdr_enabled();
    if (!gdr)
      qudaMemcpy(send_host.data_host(), send_device.data_device(), bytes, qudaMemcpyDeviceToHost);
    char *send = static_cast<char *>(gdr ? send_device.data_device() : send_host.data_host());
    char *recv = static_cast<char *>(gdr ? recv_device.data_device() : recv_host.data_host());

    std::array<MsgHandle *, region_count> send_handle = {};
    std::array<MsgHandle *, region_count> recv_handle = {};
    const size_t site_bytes = rhs_capacity * in[0].Ncolor() * 2 * static_cast<size_t>(precision);
    for (int region = 0; region < region_count; region++) {
      if (region_sites[region] == 0) continue;
      int displacement[4];
      decode_region(region, displacement);
      const size_t offset = region_offset[region] * site_bytes;
      const size_t message_bytes = region_sites[region] * site_bytes;
      recv_handle[region] = comm_declare_receive_displaced(recv + offset, displacement, message_bytes);
      for (int d = 0; d < 4; d++) displacement[d] = -displacement[d];
      send_handle[region] = comm_declare_send_displaced(send + offset, displacement, message_bytes);
      comm_start(recv_handle[region]);
    }
    for (int region = 0; region < region_count; region++)
      if (send_handle[region]) comm_start(send_handle[region]);
    for (int region = 0; region < region_count; region++) {
      if (!send_handle[region]) continue;
      comm_wait(send_handle[region]);
      comm_wait(recv_handle[region]);
      comm_free(send_handle[region]);
      comm_free(recv_handle[region]);
    }

    if (!gdr)
      qudaMemcpy(recv_device.data_device(), recv_host.data_host(), bytes, qudaMemcpyHostToDevice);
    return *this;
  }

  void StaggeredRotatingHalo::reset()
  {
    send_device = quda_ptr();
    recv_device = quda_ptr();
    send_host = quda_ptr();
    recv_host = quda_ptr();
    region_offset.fill(0);
    region_sites.fill(0);
    x.fill(0);
    r.fill(0);
    sites = 0;
    bytes = 0;
    rhs_capacity = 0;
    precision = QUDA_INVALID_PRECISION;
  }
} // namespace quda
