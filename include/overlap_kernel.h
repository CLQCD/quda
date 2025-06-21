/**
   @file overlap.h

   @section DESCRIPTION
*/

#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

namespace quda
{
  struct OverlapKernel {
    vector<ColorSpinorField> evecs;
    vector<double> evals;
    double kappa;
    double epsilon;
    vector<double> remez_tol;
    vector<vector<double>> remez_coeff;
    vector<int> remez_order;

    OverlapKernel(cvector<ColorSpinorField> &evecs, cvector<Complex> &evals, double kappa, cvector<double> remez_tol);
    OverlapKernel(const OverlapKernel *overlap_kernel, QudaPrecision precision);
    ~OverlapKernel() = default;

    inline QudaPrecision Precision() const { return evecs[0].Precision(); }
    inline double Kappa() const { return kappa; }
  };
} // namespace quda
