#ifndef QUDA_ARPACK_INTERFACE_H_
#define QUDA_ARPACK_INTERFACE_H_

#pragma once

#include <string>
#include <complex>
#include <stdio.h>

#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <vector>
#include <algorithm>

#ifdef ARPACK_LIB

#define ARPACK(s) s ## _

#ifdef __cplusplus
extern "C" {
#endif

#if (defined (QMP_COMMS) || defined (MPI_COMMS))
  
  extern int ARPACK(pcnaupd) (int *fcomm, int *ido, char *bmat, int *n, char *which,
			      int *nev, float *tol, std::complex<float> *resid,
			      int *ncv, std::complex<float> *v, int *ldv, int *iparam,
			      int *ipntr, std::complex<float> *workd,
			      std::complex<float> *workl, int *lworkl, float *rwork,
			      int *info);
  
  
  extern int ARPACK(pznaupd) (int *fcomm, int *ido, char *bmat, int *n, char *which,
			      int *nev, double *tol, std::complex<double> *resid,
			      int *ncv, std::complex<double> *v, int *ldv, int *iparam,
			      int *ipntr, std::complex<double> *workd,
			      std::complex<double> *workl, int *lworkl, double *rwork,
			      int *info);
  
  
  extern int ARPACK(pcneupd) (int *fcomm, int *comp_evecs, char *howmany, int *select,
			      std::complex<float> *evals, std::complex<float> *v,
			      int *ldv, std::complex<float> *sigma,
			      std::complex<float> *workev, char *bmat, int *n,
			      char *which, int *nev, float *tol,
			      std::complex<float> *resid, int *ncv,
			      std::complex<float> *v1, int *ldv1, int *iparam,
			      int *ipntr, std::complex<float> *workd,
			      std::complex<float> *workl, int *lworkl,
			      float *rwork, int *info);
  
  
  extern int ARPACK(pzneupd) (int *fcomm, int *comp_evecs, char *howmany, int *select,
			      std::complex<double> *evals, std::complex<double> *v,
			      int *ldv, std::complex<double> *sigma,
			      std::complex<double> *workev, char *bmat, int *n,
			      char *which, int *nev, double *tol,
			      std::complex<double> *resid, int *ncv,
			      std::complex<double> *v1, int *ldv1, int *iparam,
			      int *ipntr, std::complex<double> *workd,
			      std::complex<double> *workl, int *lworkl,
			      double *rwork, int *info);
  
#else
  
  extern int ARPACK(cnaupd) (int *ido, char *bmat, int *n, char *which, int *nev,
			     float *tol, std::complex<float> *resid, int *ncv,
			     std::complex<float> *v, int *ldv, int *iparam, int *ipntr,
			     std::complex<float> *workd, std::complex<float> *workl,
			     int *lworkl, float *rwork, int *info);
  
  
  extern int ARPACK(znaupd)(int *ido, char *bmat, int *n, char *which, int *nev,
			    double *tol, std::complex<double> *resid, int *ncv,
			    std::complex<double> *v, int *ldv, int *iparam, int *ipntr,
			    std::complex<double> *workd, std::complex<double> *workl, 
			    int *lworkl, double *rwork, int *info);
  
  
  extern int ARPACK(cneupd) (int *comp_evecs, char *howmany, int *select,
			     std::complex<float> *evals, std::complex<float> *v,
			     int *ldv, std::complex<float> *sigma,
			     std::complex<float> *workev, char *bmat, int *n,
			     char *which, int *nev, float *tol,
			     std::complex<float> *resid, int *ncv,
			     std::complex<float> *v1, int *ldv1, int *iparam,
			     int *ipntr, std::complex<float> *workd,
			     std::complex<float> *workl, int *lworkl,
			     float *rwork, int *info);			
  
  
  extern int ARPACK(zneupd) (int *comp_evecs, char *howmany, int *select,
			     std::complex<double> *evals, std::complex<double> *v,
			     int *ldv, std::complex<double> *sigma,
			     std::complex<double> *workev, char *bmat, int *n,
			     char *which, int *nev, double *tol,
			     std::complex<double> *resid, int *ncv,
			     std::complex<double> *v1, int *ldv1, int *iparam,
			     int *ipntr, std::complex<double> *workd,
			     std::complex<double> *workl, int *lworkl,
			     double *rwork, int *info);
  
#endif
  
#ifdef __cplusplus
}
#endif

#endif //ARPACK_LIB

namespace quda{
  
  /**
   *  Interface function to the external ARPACK library. This function utilizes 
   *  ARPACK's implemntation of the Implicitly Restarted Arnoldi Method to compute a 
   *  number of eigenvectors/eigenvalues with user specified features, such as those 
   *  with small real part, small magnitude etc. Parallel (OMP/MPI) version
   *  is also supported.
   *  @param[in/out] B           Container of eigenvectors
   *  @param[in/out] evals       A pointer to eigenvalue array.
   *  @param[in]     matEigen    Any QUDA implementation of the matrix-vector operation
   *  @param[in]     matPrec     Precision of the matrix-vector operation
   *  @param[in]     arpackPrec  Precision of IRAM procedures. 
   *  @param[in]     tol         tolerance for computing eigenvalues with ARPACK
   *  @param[in]     nev         number of eigenvectors 
   *  @param[in]     ncv         size of the subspace used by IRAM. 
   *                             ncv must satisfy the two inequalities 
   *                             2 <= ncv-nev and ncv <= *B[0].Length()
   *  @param[in]     target      eigenvector selection criteria:  
   *                             'LM' -> want the eigenvalues of largest mag.
   *                             'SM' -> want the eigenvalues of smallest mag.
   *                             'LR' -> want the eigenvalues of largest real part.
   *                             'SR' -> want the eigenvalues of smallest real part.
   *                             'LI' -> want the eigenvalues of largest imag part.
   *                             'SI' -> want the eigenvalues of smallest imag part.
   **/
  
  void arpackSolve(void *h_evecs, void *h_evals,
		   QudaInvertParam *inv_param,
		   QudaArpackParam *arpack_param,
		   DiracParam *d_param, int *local_dim);

  void arpackMGComparisonSolve(void *h_evecs, void *h_evals,
			       DiracMatrix &matSmooth,
			       QudaInvertParam *inv_param,
			       QudaArpackParam *arpack_param,
			       ColorSpinorParam *cpuParam);
  
}//endof namespace quda 

#endif
