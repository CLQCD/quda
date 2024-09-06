#pragma once

#ifdef OPENBLAS_LIB
#define EIGEN_USE_LAPACKE
#define EIGEN_USE_BLAS
#endif

#if defined(__NVCOMPILER) // WAR for nvc++ until we update to latest Eigen
#define EIGEN_DONT_VECTORIZE
#endif

#include <math.h>

#define GCC_COMPILER (defined(__GNUC__) && !defined(__clang__))

// hide annoying warning
#ifdef GCC_COMPILER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <Eigen/LU>

#ifdef GCC_COMPILER
#pragma GCC diagnostic pop
#endif

#undef GCC_COMPILER

using namespace Eigen;
