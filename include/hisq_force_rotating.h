#pragma once

#include <quda.h>

#ifdef __cplusplus
namespace quda
{
  class ColorSpinorField;
  class GaugeField;
  class StaggeredRotatingHalo;

  /**
      Add one packed quark's rotating contribution to the production staple
      oprod.  The extended level-2 X links and halo workspace are
      reused across rational terms in the force stage.
      @param[in,out] staple_oprod Level-2 force accumulator.
      @param[in] level2_x_extended Extended pure level-2 X links.
      @param[in] packed Full even/odd packed quark field.
      @param[in,out] halo Reusable diagonal endpoint halo.
      @param[in] angular_velocity Rotation angular velocity in lattice units.
      @param[in] coeff Rational-residue force coefficient.
   */
  void addHISQRotatingOprod(GaugeField &staple_oprod, const GaugeField &level2_x_extended,
                            ColorSpinorField &packed, StaggeredRotatingHalo &halo, double angular_velocity,
                            double coeff);

  /**
     Rotation-specific implementation behind the resident-state C adapter.
     @param[in,out] resident_momentum File-private resident momentum owner.
     @param[out] momentum Optional external momentum destination.
     @param[in] dt Molecular-dynamics force scale.
     @param[in] level2_coeff Level-2 HISQ path coefficients.
     @param[in] fat7_coeff Level-1 Fat7 path coefficients.
     @param[in] level2_fat Pure level-2 X links in external order.
     @param[in] w_link Reunitarized level-1 W links.
     @param[in] v_link Pre-reunitarization level-1 V links.
     @param[in] u_link Phased thin U links.
     @param[in] quark Packed rational-solution fields.
     @param[in] num Number of one-link rational terms.
     @param[in] num_naik Number of Naik/epsilon rational terms.
     @param[in] coeff Per-term ordinary and Naik force coefficients.
     @param[in] angular_velocity Rotation angular velocity in lattice units.
     @param[in,out] param Gauge geometry and momentum residency contract.
   */
  void computeHISQRotatingForce(GaugeField &resident_momentum, void *momentum, double dt,
                                const double level2_coeff[6], const double fat7_coeff[6],
                                const void *level2_fat, const void *w_link, const void *v_link,
                                const void *u_link, void **quark, int num, int num_naik,
                                double **coeff, double angular_velocity, QudaGaugeParam *param);
}

extern "C" {
#endif

  /**
      Compute the ordinary HISQ force plus the rotating f0 contribution, then
      propagate both through QUDA's standard HISQ smearing backward chain.
      Global x and y extents must be even; the rotation axis is fixed at the
      shift-center-half midpoint.
      @param[out] momentum Optional external momentum destination.
      @param[in] dt Molecular-dynamics force scale.
      @param[in] level2_coeff Level-2 HISQ path coefficients.
      @param[in] fat7_coeff Level-1 Fat7 path coefficients.
      @param[in] level2_fat Pure level-2 X links in external order.
      @param[in] w_link Reunitarized level-1 W links.
      @param[in] v_link Pre-reunitarization level-1 V links.
      @param[in] u_link Phased thin U links.
      @param[in] quark Packed rational-solution fields.
      @param[in] num Number of one-link rational terms.
      @param[in] num_naik Number of Naik/epsilon rational terms.
      @param[in] coeff Per-term ordinary and Naik force coefficients.
      @param[in] angular_velocity Rotation angular velocity in lattice units.
      @param[in,out] param Gauge geometry and momentum residency contract.
   */
  void computeHISQRotatingForceQuda(void *momentum, double dt, const double level2_coeff[6],
                                    const double fat7_coeff[6], const void *level2_fat, const void *w_link,
                                    const void *v_link, const void *u_link, void **quark, int num,
                                    int num_naik, double **coeff, double angular_velocity, QudaGaugeParam *param);

#ifdef __cplusplus
}
#endif
