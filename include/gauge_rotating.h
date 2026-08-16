#pragma once

#include <quda.h>

#ifdef __cplusplus
namespace quda
{
  class GaugeField;

  /**
     Internal launcher used by QUDA's resident-state force wrapper.
     @param[in,out] mom Momentum field receiving the force update.
     @param[in] u Extended, unphased thin gauge field.
     @param[in] epsilon Molecular-dynamics force scale.
     @param[in] context Opaque rotating gauge context.
   */
  void gaugeForceRotating(GaugeField &mom, const GaugeField &u, double epsilon, void *context);

  /**
      Apply the rotating gauge force using QUDA's resident thin gauge.
      The caller supplies the file-private resident momentum by reference so
      rotation code does not expose or look up ordinary QUDA state.
      @param[in,out] resident_momentum File-private resident momentum owner.
      @param[out] mom Optional external momentum destination.
      @param[in] context Opaque rotating gauge context.
      @param[in] dt Molecular-dynamics force scale.
      @param[in,out] param Gauge geometry and momentum residency contract.
      @return Zero on success.
   */
  int computeGaugeRotatingForce(GaugeField &resident_momentum, void *mom, void *context,
                                double dt, QudaGaugeParam *param);
}

extern "C" {
#endif

  /**
     Create the immutable device context used by the rotating tree-improved gauge
     action.

     Paths use QUDA's direct encoding: 0..3 are forward links and 4..7 encode
     backward links as 7-direction.  Field indices select the fixed
     shift-center-half basis {x, y, x^2, y^2, x^2+y^2, xy}; -1 selects a
     constant coefficient.  Global x and y extents must be even; the rotation
     axis is fixed at their shift-center-half midpoint.  The context validates
     loop closure, force-staple closure, scalar-halo bounds, and path lengths, then uploads all metadata
     once for reuse by every HMC energy and force evaluation.

     @param[in] local_dim Local physical lattice extents in x,y,z,t order.
     @param[in] radius Scalar-coordinate halo radius in each dimension.
     @param[in] action_path Closed action paths in direct QUDA encoding.
     @param[in] action_length Length of each action path.
     @param[in] action_coeff Scalar coefficient of each action path.
     @param[in] action_field_index Coordinate-basis component for each action path, or -1.
     @param[in] action_num_paths Number of action paths.
     @param[in] action_max_length Allocated stride of the action-path table.
     @param[in] force_path Staple paths grouped by differentiated-link direction.
     @param[in] force_length Length table for each force staple.
     @param[in] force_coeff Scalar coefficient table for each force staple.
     @param[in] force_field_index Coordinate-basis component table, or -1 entries.
     @param[in] force_field_offset Path-origin offset used to sample each coordinate coefficient.
     @param[in] force_num_paths Allocated paths per link direction.
     @param[in] force_max_length Allocated stride of each force-path table.
     @return Opaque context handle owned by the caller.
   */
  void *createGaugeRotatingContextQuda(
    const int local_dim[4], const int radius[4],
    int **action_path, const int *action_length, const double *action_coeff,
    const int *action_field_index, int action_num_paths, int action_max_length,
    int ***force_path, int **force_length, double **force_coeff,
    int **force_field_index, int ***force_field_offset, int force_num_paths,
    int force_max_length);

  /** @param[in] context Context returned by createGaugeRotatingContextQuda; nullptr is allowed. */
  void destroyGaugeRotatingContextQuda(void *context);

  /**
     Evaluate the rotating tree-improved gauge action on the resident gauge.
     @param[in] context Opaque rotating gauge context.
     @return Globally reduced, unnormalized action energy.
   */
  double computeGaugeRotatingActionQuda(void *context);

  /**
     Add the rotating tree-improved gauge force from the resident thin gauge
     to resident or returned momentum.
     @param[out] mom Optional external momentum destination; nullptr selects resident momentum only.
     @param[in] context Opaque rotating gauge context.
     @param[in] dt Molecular-dynamics force scale.
     @param[in,out] param Gauge geometry and momentum residency contract.
     @return Zero on success.
   */
  int computeGaugeRotatingForceQuda(void *mom, void *context, double dt, QudaGaugeParam *param);

#ifdef __cplusplus
}
#endif
