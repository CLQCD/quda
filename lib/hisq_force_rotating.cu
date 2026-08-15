#include <tunable_nd.h>
#include <instantiate.h>
#include <comm_quda.h>
#include <hisq_force_rotating.h>
#include <kernels/hisq_force_rotating.cuh>
#include <staggered_rotating_halo.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <ks_improved_force.h>
#include <momentum.h>
#include <quda_ptr.h>
#include <staggered_oprod.h>
#include <unitarization_links.h>

#include <cstring>
#include <cmath>
#include <limits>
#include <memory>

// Launcher for the rotating-frame fermion force contribution to the HISQ
// one-link oprod (see kernels/hisq_force_rotating.cuh).  Single-file
// TunableKernel1D wrapper. The oprod uses unreconstructed links and is
// instantiated at the precisions supported by the rotating Dslash.

namespace quda
{

  template <typename Float, int nColor> class HisqForceRotating : public TunableKernel1D
  {
    GaugeField &f0;                           /** Level-2 force accumulator updated in place. */
    const GaugeField &U;                      /** Extended pre-epsilon level-2 X links. */
    const StaggeredRotatingHalo &spinor_halo; /** Packed off-rank endpoint workspace. */
    ColorSpinorField &in;                     /** Full even/odd rational-solution field. */
    double angular_velocity;                  /** Rotation angular velocity in lattice units. */
    double coeff;                             /** Rational-residue force coefficient. */

    unsigned int minThreads() const override { return 8 * in.VolumeCB(); }

  public:
    /**
       @param[in,out] f0 Level-2 force accumulator.
       @param[in] U Extended level-2 X links.
       @param[in] spinor_halo Packed off-rank endpoint spinors.
       @param[in] in Full even/odd rational-solution field.
       @param[in] angular_velocity Rotation angular velocity in lattice units.
       @param[in] coeff Rational-residue force coefficient.
     */
    HisqForceRotating(GaugeField &f0, const GaugeField &U, const StaggeredRotatingHalo &spinor_halo,
                      ColorSpinorField &in, double angular_velocity, double coeff) :
      TunableKernel1D(in),
      f0(f0),
      U(U),
      spinor_halo(spinor_halo),
      in(in),
      angular_velocity(angular_velocity),
      coeff(coeff)
    {
      // The source-scatter predecessor used a different problem size under
      // the old key.  Keep a distinct key so a persisted tune cache can never
      // reuse its VolumeCB grid for this 8*VolumeCB link-owned launch.
      strcat(aux, ",hisq-rotating-oprod-link-owned-v1");
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    /** @param[in] stream Device stream used for the link-owned oprod launch. */
    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<HisqForceRotatingLinkOprod>(
        tp, stream,
        HisqForceRotatingArg<Float, nColor>(f0, U, spinor_halo, in, angular_velocity, coeff));
    }

    // The kernel adds to f0, so tuning trials must preserve the input field.
    void preTune() override { f0.backup(); }
    void postTune() override { f0.restore(); }

    long long flops() const override
    {
      long long Nc = nColor;
      // 72 paths/odd site x 3 owned-link occurrences.
      long long per_deposit = 2 * (8 * Nc * Nc) + 2 * Nc * Nc;
      return in.VolumeCB() * 72 * 3 * per_deposit;
    }

    long long bytes() const override
    {
      long long per_site = U.Bytes() / U.Volume() * 72 * 2 + in.Bytes() / in.Volume() * 72 * 2
        + f0.Bytes() / f0.Volume() * 8;
      return in.VolumeCB() * per_site;
    }
  };

  namespace
  {
    /**
       @param[in] f0 Level-2 force accumulator.
       @param[in] fat Extended level-2 X links.
       @param[in] packed Full even/odd rational-solution field.
     */
    void checkRotatingOprodFields(const GaugeField &f0, const GaugeField &fat, const ColorSpinorField &packed)
    {
      checkLocation(f0, fat, packed);
      checkPrecision(f0, fat, packed);
      if (f0.Precision() != QUDA_DOUBLE_PRECISION)
        errorQuda("Rotating HISQ oprod only supports double precision, got %d", f0.Precision());
      if (f0.Reconstruct() != QUDA_RECONSTRUCT_NO)
        errorQuda("Rotating HISQ oprod must use QUDA_RECONSTRUCT_NO for f0, got %d", f0.Reconstruct());
      if (fat.Reconstruct() != QUDA_RECONSTRUCT_NO)
        errorQuda("Rotating HISQ oprod must use QUDA_RECONSTRUCT_NO for fat links, got %d", fat.Reconstruct());
      if (f0.Ncolor() != 3 || fat.Ncolor() != 3) errorQuda("Rotating HISQ oprod only supports Ncolor=3");
      if (packed.Nspin() != 1 || packed.Ncolor() != 3)
        errorQuda("Rotating HISQ oprod requires a packed Nspin=1, Ncolor=3 field");
      if (packed.SiteSubset() != QUDA_FULL_SITE_SUBSET || packed.SiteOrder() != QUDA_EVEN_ODD_SITE_ORDER)
        errorQuda("Rotating HISQ oprod requires a full-site field with even-odd ordering");
    }

    /**
       @param[in,out] f0 Level-2 force accumulator.
       @param[in] fat_extended Extended level-2 X links.
       @param[in] packed Full even/odd rational-solution field.
       @param[in,out] halo_owner Reusable diagonal spinor halo.
       @param[in] angular_velocity Rotation angular velocity in lattice units.
       @param[in] coeff Rational-residue force coefficient.
     */
    void launchRotatingOprod(GaugeField &f0, const GaugeField &fat_extended, ColorSpinorField &packed,
                             StaggeredRotatingHalo &halo_owner, double angular_velocity, double coeff)
    {
      for (int d = 0; d < 2; d++) {
        const int global_dim = packed.X()[d] * comm_dim(d);
        if (global_dim % 2 != 0)
          errorQuda("Rotating HISQ force requires an even global extent in direction %d, got %d",
                    d, global_dim);
      }
      const bool use_halo = comm_dim_partitioned(0) || comm_dim_partitioned(1) || comm_dim_partitioned(3);
      StaggeredRotatingHalo empty_halo;
      const StaggeredRotatingHalo *spinor_halo = &empty_halo;
      if (use_halo) {
        cvector_ref<const ColorSpinorField> packed_ref {packed};
        spinor_halo = &halo_owner.exchange(packed_ref, QUDA_INVALID_PARITY);
      }
      HisqForceRotating<double, 3>(f0, fat_extended, *spinor_halo, packed, angular_velocity, coeff);
    }
  } // namespace

  void addHISQRotatingOprod(GaugeField &staple_oprod, const GaugeField &fat_extended,
                            ColorSpinorField &packed, StaggeredRotatingHalo &halo, double angular_velocity,
                            double coeff)
  {
    if (std::abs(angular_velocity) <= std::numeric_limits<double>::epsilon()) return;
    checkRotatingOprodFields(staple_oprod, fat_extended, packed);
    launchRotatingOprod(staple_oprod, fat_extended, packed, halo, angular_velocity, coeff);
  }

  void computeHISQRotatingForce(GaugeField &resident_momentum, void *const milc_momentum, double dt,
                                const double level2_coeff[6], const double fat7_coeff[6],
                                const void *const level2_fat, const void *const w_link,
                                const void *const v_link, const void *const u_link, void **fermion,
                                int num_terms, int num_naik_terms, double **coeff, double angular_velocity,
                                QudaGaugeParam *gParam)
  {
    using namespace fermion_force;

    {
      // default settings for the unitarization
      const double unitarize_eps = 1e-14;
      const double hisq_force_filter = 5e-5;
      const double max_det_error = 1e-10;
      const bool   allow_svd = true;
      const bool   svd_only = false;
      const double svd_rel_err = 1e-8;
      const double svd_abs_err = 1e-8;

      setUnitarizeForceConstants(unitarize_eps, hisq_force_filter, max_det_error, allow_svd, svd_only, svd_rel_err, svd_abs_err);
    }

    // Save input reconstruct type (applied to W and U fields) and set
    // the reconstruct type to QUDA_RECONSTRUCT_NO
    QudaReconstructType cuda_link_recon = gParam->reconstruct;
    gParam->reconstruct = QUDA_RECONSTRUCT_NO;

    // Create a copy of the setup for the gauge links
    QudaGaugeParam gParam_field;
    memcpy(&gParam_field, gParam, sizeof(QudaGaugeParam));

    // Check reconstruct
    if (cuda_link_recon == QUDA_RECONSTRUCT_9) {
      warningQuda("Attempting to use recon 9 for HISQ force. Resetting to 13...");
      cuda_link_recon = QUDA_RECONSTRUCT_13;
    }

    if (cuda_link_recon != QUDA_RECONSTRUCT_NO && cuda_link_recon != QUDA_RECONSTRUCT_13)
      errorQuda("Invalid reconstruct %d", cuda_link_recon);

    logQuda(QUDA_VERBOSE, "Reconstruct type for HISQ force: %d\n", cuda_link_recon);

    // create the device outer-product field
    GaugeFieldParam oParam(*gParam);
    oParam.location = QUDA_CUDA_FIELD_LOCATION;
    oParam.nFace = 0;
    oParam.create = QUDA_ZERO_FIELD_CREATE;
    oParam.link_type = QUDA_GENERAL_LINKS;
    oParam.reconstruct = QUDA_RECONSTRUCT_NO;
    oParam.setPrecision(gParam->cpu_prec, true);
    oParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

    GaugeField stapleOprod(oParam);
    GaugeField oneLinkOprod(oParam);
    GaugeField naikOprod(oParam);

    // The action owns the pure level-2 X link.  Force consumes it explicitly
    // instead of consulting the Dslash resident-link registry.
    gParam_field.type = QUDA_GENERAL_LINKS;
    gParam_field.t_boundary = QUDA_ANTI_PERIODIC_T;
    gParam_field.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
    gParam_field.staggered_phase_applied = true;
    gParam_field.gauge_fix = QUDA_GAUGE_FIXED_NO;

    GaugeFieldParam xParam(gParam_field);
    xParam.location = QUDA_CPU_FIELD_LOCATION;
    xParam.create = QUDA_REFERENCE_FIELD_CREATE;
    xParam.link_type = QUDA_GENERAL_LINKS;
    xParam.reconstruct = QUDA_RECONSTRUCT_NO;
    xParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    xParam.gauge = const_cast<void *>(level2_fat);
    GaugeField cpuXLink(xParam);

    xParam.location = QUDA_CUDA_FIELD_LOCATION;
    xParam.create = QUDA_NULL_FIELD_CREATE;
    xParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    xParam.setPrecision(gParam->cpu_prec, true);
    GaugeField localXLink(xParam);
    localXLink.copy(cpuXLink);
    const lat_dim_t xRadius = {2, 2, comm_dim_partitioned(2) ? 1 : 0, 1};
    std::unique_ptr<GaugeField> cudaXLink(createExtendedGauge(localXLink, xRadius, getProfile(), true));

    double act_path_coeff[6] = {0, 1, level2_coeff[2], level2_coeff[3], level2_coeff[4], level2_coeff[5]};
    // You have to look at the MILC routine to understand the following
    // Basically, I have already absorbed the one-link coefficient

    { // do outer-product computation
      ColorSpinorParam qParam;
      qParam.nColor = 3;
      qParam.nSpin = 1;
      qParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      qParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
      qParam.nDim = 4;
      qParam.pc_type = QUDA_4D_PC;
      qParam.setPrecision(oParam.Precision(), oParam.Precision(), true);
      qParam.pad = 0;
      for (int dir=0; dir<4; ++dir) qParam.x[dir] = oParam.x[dir];

      // create the device quark field
      qParam.create = QUDA_NULL_FIELD_CREATE;
      qParam.location = QUDA_CUDA_FIELD_LOCATION;
      ColorSpinorField cudaQuark(qParam);

      StaggeredRotatingHalo rotating_halo;

      // create the host quark field
      qParam.location = QUDA_CPU_FIELD_LOCATION;
      qParam.create = QUDA_REFERENCE_FIELD_CREATE;
      qParam.fieldOrder = QUDA_SPACE_COLOR_SPIN_FIELD_ORDER;
      qParam.v = fermion[0];

      { // regular terms
        GaugeField *oprod[2] = {&stapleOprod, &naikOprod};

        // loop over different quark fields
        for (int i = 0; i < num_terms; ++i) {

          // Wrap the MILC quark field
          qParam.v = fermion[i];
          ColorSpinorField cpuQuark(qParam); // create host quark field

          cudaQuark = cpuQuark;
          computeStaggeredOprod(oprod, cudaQuark, coeff[i], 3);
          addHISQRotatingOprod(stapleOprod, *cudaXLink, cudaQuark, rotating_halo, angular_velocity,
              coeff[i][0]);
        }
      }

      { // naik terms
        oneLinkOprod.copy(stapleOprod, level2_coeff[0]);
        GaugeField *oprod[2] = {&oneLinkOprod, &naikOprod};

        // loop over different quark fields
        for (int i = 0; i < num_naik_terms; ++i) {

          // Wrap the MILC quark field
          qParam.v = fermion[i + num_terms - num_naik_terms];
          ColorSpinorField cpuQuark(qParam); // create host quark field

          cudaQuark = cpuQuark;
          computeStaggeredOprod(oprod, cudaQuark, coeff[i + num_terms], 3);
        }
      }
    }

    // Copy outer product fields into input force fields
    oParam.create = QUDA_NULL_FIELD_CREATE;
    oParam.nFace = 1;
    oParam.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
    lat_dim_t R = {2 * comm_dim_partitioned(0), 2 * comm_dim_partitioned(1), 2 * comm_dim_partitioned(2),
      2 * comm_dim_partitioned(3)};
    for (int dir = 0; dir < 4; ++dir) {
      oParam.x[dir] += 2 * R[dir];
      oParam.r[dir] = R[dir];
    }

    GaugeField cudaInForce(oParam);
    copyExtendedGauge(cudaInForce, stapleOprod, QUDA_CUDA_FIELD_LOCATION);
    stapleOprod = GaugeField();

    GaugeField cudaOutForce(oParam);
    copyExtendedGauge(cudaOutForce, oneLinkOprod, QUDA_CUDA_FIELD_LOCATION);
    oneLinkOprod = GaugeField();

    // Create CPU momentum fields, prepare GPU momentum param
    GaugeFieldParam param(*gParam);
    param.location = QUDA_CPU_FIELD_LOCATION;
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    param.link_type = QUDA_ASQTAD_MOM_LINKS;
    param.reconstruct = QUDA_RECONSTRUCT_10;
    param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    param.gauge = milc_momentum;
    GaugeField cpuMom = (!gParam->use_resident_mom) ? GaugeField(param) : GaugeField();

    param.location = QUDA_CUDA_FIELD_LOCATION;
    param.create = QUDA_ZERO_FIELD_CREATE;
    param.setPrecision(param.Precision(), true);
    GaugeFieldParam momParam(param);

    // Create CPU W, V, and U fields
    gParam_field.type = QUDA_GENERAL_LINKS;
    gParam_field.t_boundary = QUDA_ANTI_PERIODIC_T;
    gParam_field.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
    gParam_field.staggered_phase_applied = true;
    gParam_field.gauge_fix = QUDA_GAUGE_FIXED_NO;

    GaugeFieldParam wParam(gParam_field);
    wParam.location = QUDA_CPU_FIELD_LOCATION;
    wParam.create = QUDA_REFERENCE_FIELD_CREATE;
    wParam.link_type = QUDA_GENERAL_LINKS;
    wParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    wParam.gauge = (void *)w_link;
    GaugeField cpuWLink(wParam);

    GaugeFieldParam vParam(wParam);
    vParam.gauge = (void *)v_link;
    GaugeField cpuVLink(vParam);

    GaugeFieldParam uParam(vParam);
    uParam.gauge = (void *)u_link;
    GaugeField cpuULink(uParam);

    // Load the W field, which contains U(3) matrices, to the device
    wParam = GaugeFieldParam(gParam_field);
    for (int dir = 0; dir < 4; dir++) {
      wParam.x[dir] += 2 * R[dir];
      wParam.r[dir] = R[dir];
    }
    wParam.location = QUDA_CUDA_FIELD_LOCATION;
    wParam.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
    wParam.reconstruct = cuda_link_recon;
    wParam.create = QUDA_NULL_FIELD_CREATE;
    wParam.setPrecision(gParam->cpu_prec, true);

    GaugeField cudaWLink(wParam);

    cudaWLink.copy(cpuWLink);

    cudaWLink.exchangeExtendedGhost(cudaWLink.R(), getProfile());

    cudaInForce.exchangeExtendedGhost(R, getProfile());
    cudaWLink.exchangeExtendedGhost(cudaWLink.R(), getProfile());
    cudaOutForce.exchangeExtendedGhost(R, getProfile());

    // Compute level two term
    hisqStaplesForce(cudaOutForce, cudaInForce, cudaWLink, act_path_coeff);

    // Load naik outer product
    copyExtendedGauge(cudaInForce, naikOprod, QUDA_CUDA_FIELD_LOCATION);
    cudaInForce.exchangeExtendedGhost(cudaWLink.R(), getProfile());
    naikOprod = GaugeField();

    // Compute Naik three-link term contribution
    hisqLongLinkForce(cudaOutForce, cudaInForce, cudaWLink, act_path_coeff[1]);

    cudaOutForce.exchangeExtendedGhost(R, getProfile());

    // Load the V field, which contains general matrices, to the device
    cudaWLink = GaugeField();

    for (int dir = 0; dir < 4; ++dir) {
      vParam.x[dir] += 2 * R[dir];
      vParam.r[dir] = R[dir];
    }
    vParam.location = QUDA_CUDA_FIELD_LOCATION;
    vParam.link_type = QUDA_GENERAL_LINKS;
    vParam.reconstruct = QUDA_RECONSTRUCT_NO;
    vParam.create = QUDA_NULL_FIELD_CREATE;
    vParam.setPrecision(gParam->cpu_prec, true);
    vParam.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
    GaugeField cudaVLink(vParam);

    cudaVLink.copy(cpuVLink);
    cudaVLink.exchangeExtendedGhost(cudaVLink.R(), getProfile());

    quda_ptr failures(QUDA_MEMORY_HOST_PINNED, sizeof(int));
    auto *num_failures_h = static_cast<int *>(failures.data_host());
    auto *num_failures_d = static_cast<int *>(failures.data_device());
    *num_failures_h = 0;
    unitarizeForce(cudaInForce, cudaOutForce, cudaVLink, num_failures_d);

    if (*num_failures_h>0) errorQuda("Error in the unitarization component of the hisq fermion force: %d failures\n", *num_failures_h);

    // Load the U field, which contains U(3) matrices, to the device
    // TODO: in theory these should just be SU(3) matrices with MILC phases?
    cudaVLink = GaugeField();

    for (int dir = 0; dir < 4; ++dir) {
      uParam.x[dir] += 2 * R[dir];
      uParam.r[dir] = R[dir];
    }
    uParam.location = QUDA_CUDA_FIELD_LOCATION;
    uParam.link_type = QUDA_GENERAL_LINKS;
    uParam.reconstruct = cuda_link_recon;
    uParam.create = QUDA_NULL_FIELD_CREATE;
    uParam.setPrecision(gParam->cpu_prec, true);
    uParam.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
    GaugeField cudaULink(uParam);

    cudaULink.copy(cpuULink);
    cudaULink.exchangeExtendedGhost(cudaULink.R(), getProfile());

    // Compute Fat7-staple term
    cudaOutForce.zero();
    hisqStaplesForce(cudaOutForce, cudaInForce, cudaULink, fat7_coeff);

    cudaInForce = GaugeField();

    hisqCompleteForce(cudaOutForce, cudaULink);

    if (gParam->use_resident_mom && !resident_momentum.Length()) errorQuda("No resident momentum field to use");
    GaugeField mom = gParam->use_resident_mom ? resident_momentum.create_alias() : GaugeField(momParam);
    updateMomentum(mom, dt, cudaOutForce, "hisq");

    // Close the paths, make anti-hermitian, and store in compressed format
    if (gParam->return_result_mom) cpuMom.copy(mom);

    if (gParam->make_resident_mom && !gParam->use_resident_mom)
      std::exchange(resident_momentum, mom);
    else if (!gParam->make_resident_mom)
      resident_momentum = GaugeField();
  }

} // namespace quda
