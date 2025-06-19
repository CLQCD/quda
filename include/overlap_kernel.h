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
    std::vector<ColorSpinorField> evecs;
    std::vector<double> evals;
    double kappa;
    double epsilon;
    std::vector<double> remez_tol;
    std::vector<std::vector<double>> remez_coeff;
    std::vector<int> remez_order;

    OverlapKernel(std::vector<ColorSpinorField> &evecs, std::vector<Complex> &evals, double kappa,
                  std::vector<double> remez_tol);
    OverlapKernel(OverlapKernel *overlap_kernel, QudaPrecision precision);
    ~OverlapKernel() = default;

    inline QudaPrecision Precision() const { return evecs[0].Precision(); }
    inline double Kappa() const { return kappa; }
  };
} // namespace quda
