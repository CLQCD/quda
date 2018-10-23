#ifndef _QUDA_INTERNAL_H
#define _QUDA_INTERNAL_H

#include <quda_cuda_api.h>
#include <string>
#include <complex>
#include <vector>

#if ((defined(QMP_COMMS) || defined(MPI_COMMS)) && !defined(MULTI_GPU))
#error "MULTI_GPU must be enabled to use MPI or QMP"
#endif

#if (!defined(QMP_COMMS) && !defined(MPI_COMMS) && defined(MULTI_GPU))
#error "MPI or QMP must be enabled to use MULTI_GPU"
#endif

#ifdef QMP_COMMS
#include <qmp.h>
#endif

#ifdef PTHREADS
#include <pthread.h>
#endif

#define TEX_ALIGN_REQ (512*2) //Fermi, factor 2 comes from even/odd
#define ALIGNMENT_ADJUST(n) ( (n+TEX_ALIGN_REQ-1)/TEX_ALIGN_REQ*TEX_ALIGN_REQ)
#include <enum_quda.h>
#include <quda.h>
#include <util_quda.h>
#include <malloc_quda.h>
#include <object.h>

// Use bindless texture on Kepler
#if (__COMPUTE_CAPABILITY__ >= 300 || __CUDA_ARCH__ >= 300)
#define USE_TEXTURE_OBJECTS
#endif

// if not using texture objects then we need to disable multi-blas support since these don't work with texture references
#ifndef USE_TEXTURE_OBJECTS
#undef MAX_MULTI_BLAS_N
#define MAX_MULTI_BLAS_N 1
#endif

#ifdef __cplusplus
extern "C" {
#endif
  
  struct QUDA_DiracField{
    void *field; /**< Pointer to a ColorSpinorField */
  };

  extern cudaDeviceProp deviceProp;  
  extern cudaStream_t *streams;
 
#ifdef PTHREADS
  extern pthread_mutex_t pthread_mutex;
#endif
 
#ifdef __cplusplus
}
#endif

namespace quda {

  typedef std::complex<double> Complex;

  /**
   * Traits for determining the maximum and inverse maximum
   * value of a (signed) char and short. Relevant for
   * fixed precision types. 
   */
  template< typename T > struct fixedMaxValue{ static constexpr float value = 0.0f; };
  template<> struct fixedMaxValue<short>{ static constexpr float value = 32767.0f; };
  template<> struct fixedMaxValue<short2>{ static constexpr float value = 32767.0f; };
  template<> struct fixedMaxValue<short4>{ static constexpr float value = 32767.0f; };
  template<> struct fixedMaxValue<char>{ static constexpr float value = 127.0f; };
  template<> struct fixedMaxValue<char2>{ static constexpr float value = 127.0f; };
  template<> struct fixedMaxValue<char4>{ static constexpr float value = 127.0f; };

  template< typename T > struct fixedInvMaxValue{ static constexpr double value = 3.402823e+38; };
  template<> struct fixedInvMaxValue<short>{ static constexpr double value = 3.051850948e-5; };
  template<> struct fixedInvMaxValue<short2>{ static constexpr double value = 3.051850948e-5; };
  template<> struct fixedInvMaxValue<short4>{ static constexpr double value = 3.051850948e-5; };
  template<> struct fixedInvMaxValue<char>{ static constexpr double value = 7.87401574e-3; };
  template<> struct fixedInvMaxValue<char2>{ static constexpr double value = 7.87401574e-3; };
  template<> struct fixedInvMaxValue<char4>{ static constexpr double value = 7.87401574e-3; };

#ifdef PTHREADS
  const int Nstream = 10;
#else
  const int Nstream = 9;
#endif

  /**
   * Check that the resident gauge field is compatible with the requested inv_param
   * @param inv_param   Contains all metadata regarding host and device storage
   */
  bool canReuseResidentGauge(QudaInvertParam *inv_param);

}

#include <timer.h>


#endif // _QUDA_INTERNAL_H
