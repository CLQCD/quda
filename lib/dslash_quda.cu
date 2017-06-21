#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>

#include <color_spinor_field.h>
#include <clover_field.h>

// these control the Wilson-type actions
#ifdef GPU_WILSON_DIRAC
//#define DIRECT_ACCESS_LINK
//#define DIRECT_ACCESS_WILSON_SPINOR
//#define DIRECT_ACCESS_WILSON_ACCUM
//#define DIRECT_ACCESS_WILSON_INTER
//#define DIRECT_ACCESS_WILSON_PACK_SPINOR
//#define DIRECT_ACCESS_CLOVER
#endif // GPU_WILSON_DIRAC

//these are access control for staggered action
#ifdef GPU_STAGGERED_DIRAC
#if (__COMPUTE_CAPABILITY__ >= 300) // Kepler works best with texture loads only
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
//#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#else // fermi
//#define DIRECT_ACCESS_FAT_LINK
//#define DIRECT_ACCESS_LONG_LINK
#define DIRECT_ACCESS_SPINOR
//#define DIRECT_ACCESS_ACCUM
//#define DIRECT_ACCESS_INTER
//#define DIRECT_ACCESS_PACK
#endif
#endif // GPU_STAGGERED_DIRAC

#include <quda_internal.h>
#include <dslash_quda.h>
#include <sys/time.h>
#include <blas_quda.h>
#include <face_quda.h>

#include <inline_ptx.h>

namespace quda {

  namespace dslash_aux {
#include <dslash_constants.h>
#include <dslash_textures.h>
#include <dslash_index.cuh>
#include <io_spinor.h>

#include <tm_core.h>              // solo twisted mass kernel
#include <tmc_core.h>              // solo twisted mass kernel
#include <clover_def.h>           // kernels for applying the clover term alone
  }

#ifndef DSLASH_SHARED_FLOATS_PER_THREAD
#define DSLASH_SHARED_FLOATS_PER_THREAD 0
#endif

#ifndef CLOVER_SHARED_FLOATS_PER_THREAD
#define CLOVER_SHARED_FLOATS_PER_THREAD 0
#endif

#ifndef NDEGTM_SHARED_FLOATS_PER_THREAD
#define NDEGTM_SHARED_FLOATS_PER_THREAD 0
#endif

  // these should not be namespaced!!
  // determines whether the temporal ghost zones are packed with a gather kernel,
  // as opposed to multiple calls to cudaMemcpy()
  static bool kernelPackT = false;

  void setKernelPackT(bool packT) { kernelPackT = packT; }

  bool getKernelPackT() { return kernelPackT; }

  namespace dslash {
    int it = 0;

#ifdef PTHREADS
    cudaEvent_t interiorDslashEnd;
#endif
    cudaEvent_t packEnd[Nstream];
    cudaEvent_t gatherStart[Nstream];
    cudaEvent_t gatherEnd[Nstream];
    cudaEvent_t scatterStart[Nstream];
    cudaEvent_t scatterEnd[Nstream];
    cudaEvent_t dslashStart;
    cudaEvent_t dslashEnd;

    // FIX this is a hack from hell
    // Auxiliary work that can be done while waiting on comms to finis
    Worker *aux_worker;

#if CUDA_VERSION >= 8000
    cuuint32_t *commsEnd_h;
    CUdeviceptr commsEnd_d[Nstream];
#endif
  }

  void createDslashEvents()
  {
    using namespace dslash;
    // add cudaEventDisableTiming for lower sync overhead
    for (int i=0; i<Nstream; i++) {
      cudaEventCreate(&packEnd[i], cudaEventDisableTiming);
      cudaEventCreate(&gatherStart[i], cudaEventDisableTiming);
      cudaEventCreate(&gatherEnd[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&scatterStart[i], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&scatterEnd[i], cudaEventDisableTiming);
    }
    cudaEventCreateWithFlags(&dslashStart, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&dslashEnd, cudaEventDisableTiming);
#ifdef PTHREADS
    cudaEventCreateWithFlags(&interiorDslashEnd, cudaEventDisableTiming);
#endif

    aux_worker = NULL;

#if CUDA_VERSION >= 8000
    commsEnd_h = static_cast<cuuint32_t*>(mapped_malloc(Nstream*sizeof(int)));
    for (int i=0; i<Nstream; i++) {
      cudaHostGetDevicePointer((void**)&commsEnd_d[i], commsEnd_h+i, 0);
      commsEnd_h[i] = 0;
    }
#endif

    checkCudaError();
  }


  void destroyDslashEvents()
  {
    using namespace dslash;

#if CUDA_VERSION >= 8000
    host_free(commsEnd_h);
    commsEnd_h = 0;
#endif

    for (int i=0; i<Nstream; i++) {
      cudaEventDestroy(packEnd[i]);
      cudaEventDestroy(gatherStart[i]);
      cudaEventDestroy(gatherEnd[i]);
      cudaEventDestroy(scatterStart[i]);
      cudaEventDestroy(scatterEnd[i]);
    }

    cudaEventDestroy(dslashStart);
    cudaEventDestroy(dslashEnd);
#ifdef PTHREADS
    cudaEventDestroy(interiorDslashEnd);
#endif

    checkCudaError();
  }

  using namespace dslash_aux;

#include <gamma5.h>		// g5 kernel
  
  /**
     Class for the gamma5 kernels, sFloat is the typename of the spinor components (double2, float4...)
  */

  template <typename sFloat>
  class Gamma5Cuda : public Tunable {
    
  private:
    cudaColorSpinorField *out;		//Output spinor
    const cudaColorSpinorField *in;		//Input spinor
    
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return in->X(0) * in->X(1) * in->X(2) * in->X(3); }
    
    char *saveOut, *saveOutNorm;
    
  public:
    Gamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in) :
      out(out), in(in) {
      bindSpinorTex<sFloat>(in, out);
      dslashParam.out = out->V();
      dslashParam.outNorm = (float*)out->Norm();
      dslashParam.in = (void*)in->V();
      dslashParam.inNorm = (float*)in->Norm();
      dslashParam.sp_stride = in->Stride();
      strcpy(aux,"gamma5");
    }
    
    virtual ~Gamma5Cuda() { unbindSpinorTex<sFloat>(in, out); }
    
    TuneKey tuneKey() const
    {
      return TuneKey(in->VolString(), typeid(*this).name());
    }
    
    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (in->Precision() == QUDA_DOUBLE_PRECISION) {
	gamma5DKernel<<<tp.grid, tp.block, tp.shared_bytes>>> (dslashParam);
      } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
	gamma5SKernel<<<tp.grid, tp.block, tp.shared_bytes>>> (dslashParam);
      } else {
	errorQuda("Undefined for precision %d", in->Precision());
      }
    }
    
    void preTune()
    {
      saveOut = new char[out->Bytes()];
      cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);

      if (typeid(sFloat) == typeid(short4))
	{
	  saveOutNorm = new char[out->NormBytes()];
	  cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
	}
    }
    
    void postTune()
    {
      cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
      delete[] saveOut;
      
      if (typeid(sFloat) == typeid(short4))
	{
	  cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
	  delete[] saveOutNorm;
	}
    }
    
    long long flops() const { return 12ll * in->VolumeCB(); }
    long long bytes() const { return in->Bytes() + in->NormBytes() + out->Bytes() + out->NormBytes(); }
  };

  /**
     Applies a gamma5 matrix to a spinor, this is the function to be called in interfaces and it requires only
     pointers to the output spinor (out) and the input spinor (in), in that order
  */

  void gamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in)
  {
    dslashParam.threads = in->Volume();

    Tunable *gamma5 = 0;

    if (in->Precision() == QUDA_DOUBLE_PRECISION) {
      gamma5 = new Gamma5Cuda<double2>(out, in);
    } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
      gamma5 = new Gamma5Cuda<float4>(out, in);
    } else if (in->Precision() == QUDA_HALF_PRECISION) {
      errorQuda("Half precision not supported for gamma5 kernel yet");
    }

    gamma5->apply(streams[Nstream-1]);
    checkCudaError();

    delete gamma5;
  }

template <typename sFloat, typename cFloat>
class CloverCuda : public Tunable {
  private:
    cudaColorSpinorField *out;
    float *outNorm;
    char *saveOut, *saveOutNorm;
    const cFloat *clover;
    const float *cloverNorm;
    const cudaColorSpinorField *in;

  protected:
    unsigned int sharedBytesPerThread() const
    {
      int reg_size = (typeid(sFloat)==typeid(double2) ? sizeof(double) : sizeof(float));
      return CLOVER_SHARED_FLOATS_PER_THREAD * reg_size;
    }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return in->VolumeCB(); }

  public:
    CloverCuda(cudaColorSpinorField *out, const cFloat *clover, const float *cloverNorm, 
	       int cl_stride, const cudaColorSpinorField *in)
      : out(out), clover(clover), cloverNorm(cloverNorm), in(in)
    {
      bindSpinorTex<sFloat>(in);

      dslashParam.out = (void*)out->V();
      dslashParam.outNorm = (float*)out->Norm();
      dslashParam.in = (void*)in->V();
      dslashParam.inNorm = (float*)in->Norm();
      dslashParam.clover = (void*)clover;
      dslashParam.cloverNorm = (float*)cloverNorm;

      dslashParam.sp_stride = in->Stride();
#ifdef GPU_CLOVER_DIRAC
      dslashParam.cl_stride = cl_stride;
#endif
    }
    virtual ~CloverCuda() { unbindSpinorTex<sFloat>(in); }
    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
      if (typeid(sFloat)==typeid(double2)) {
	cloverDKernel <<<gridDim, tp.block, tp.shared_bytes, stream>>>(dslashParam);
      } else if (typeid(sFloat)==typeid(float4)) {
	cloverSKernel <<<gridDim, tp.block, tp.shared_bytes, stream>>>(dslashParam);
      } else {
	cloverHKernel <<<gridDim, tp.block, tp.shared_bytes, stream>>>(dslashParam);
      }
    }
    virtual TuneKey tuneKey() const { return TuneKey(in->VolString(), typeid(*this).name()); }

    // Need to save the out field if it aliases the in field
    void preTune() {
      if (in == out) {
        saveOut = new char[out->Bytes()];
        cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);
        if (typeid(sFloat) == typeid(short4)) {
          saveOutNorm = new char[out->NormBytes()];
          cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
        }
      }
    }

    // Restore if the in and out fields alias
    void postTune() {
      if (in == out) {
        cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
        delete[] saveOut;
        if (typeid(sFloat) == typeid(short4)) {
          cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
          delete[] saveOutNorm;
        }
      }
    }

    long long flops() const { return 504ll * in->VolumeCB(); }
};


void cloverCuda(cudaColorSpinorField *out, const cudaGaugeField &gauge, const FullClover clover, 
		const cudaColorSpinorField *in, const int parity) {

  dslashParam.parity = parity;
  dslashParam.threads = in->Volume();

#ifdef GPU_CLOVER_DIRAC
  Tunable *clov = 0;
  void *cloverP, *cloverNormP;
  QudaPrecision clover_prec = bindCloverTex(clover, parity, &cloverP, &cloverNormP);

  if (in->Precision() != clover_prec)
    errorQuda("Mixing clover and spinor precision not supported");

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
    clov = new CloverCuda<double2, double2>(out, (double2*)cloverP, (float*)cloverNormP, clover.stride, in);
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    clov = new CloverCuda<float4, float4>(out, (float4*)cloverP, (float*)cloverNormP, clover.stride, in);
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    clov = new CloverCuda<short4, short4>(out, (short4*)cloverP, (float*)cloverNormP, clover.stride, in);
  }
  clov->apply(0);

  unbindCloverTex(clover);
  checkCudaError();

  delete clov;
#else
  errorQuda("Clover dslash has not been built");
#endif
}


template <typename sFloat>
class TwistGamma5Cuda : public Tunable {

  private:
    cudaColorSpinorField *out;
    const cudaColorSpinorField *in;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return in->X(0) * in->X(1) * in->X(2) * in->X(3); }

    char *saveOut, *saveOutNorm;

  public:
    TwistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
        double kappa, double mu, double epsilon, const int dagger, QudaTwistGamma5Type twist) :
      out(out), in(in) 
  {
    bindSpinorTex<sFloat>(in);
    dslashParam.out = (void*)out->V();
    dslashParam.outNorm = (float*)out->Norm();
    dslashParam.in = (void*)in->V();
    dslashParam.inNorm = (float*)in->Norm();
    dslashParam.sp_stride = in->Stride();
    if(in->TwistFlavor() == QUDA_TWIST_SINGLET) {
      setTwistParam(dslashParam.a, dslashParam.b, kappa, mu, dagger, twist);
      dslashParam.c = 0.0;
#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
      dslashParam.fl_stride = in->VolumeCB();
#endif
    } else {//twist doublet
      dslashParam.a = kappa, dslashParam.b = mu, dslashParam.c = epsilon;
#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
      dslashParam.fl_stride = in->VolumeCB()/2;
#endif
    }
    dslashParam.a_f = dslashParam.a;
    dslashParam.b_f = dslashParam.b;
    dslashParam.c_f = dslashParam.c;
  }

    virtual ~TwistGamma5Cuda() { unbindSpinorTex<sFloat>(in); }

    TuneKey tuneKey() const { return TuneKey(in->VolString(), typeid(*this).name(), in->AuxString()); }

    void apply(const cudaStream_t &stream) 
    {
#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
      if(in->TwistFlavor() == QUDA_TWIST_SINGLET) {
        twistGamma5Kernel<sFloat,false><<<gridDim, tp.block, tp.shared_bytes, stream>>>(dslashParam);
      } else {
        twistGamma5Kernel<sFloat,true><<<gridDim, tp.block, tp.shared_bytes, stream>>>(dslashParam);
      }
#endif
    }

    void preTune() {
      saveOut = new char[out->Bytes()];
      cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short4)) {
        saveOutNorm = new char[out->NormBytes()];
        cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
      }
    }

    void postTune() {
      cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short4)) {
        cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
        delete[] saveOutNorm;
      }
    }

    long long flops() const { return 24ll * in->VolumeCB(); }
    long long bytes() const { return in->Bytes() + in->NormBytes() + out->Bytes() + out->NormBytes(); }
};

//!ndeg tm: 
void twistGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
    const int dagger, const double &kappa, const double &mu, const double &epsilon,   const QudaTwistGamma5Type twist)
{
  if(in->TwistFlavor() == QUDA_TWIST_SINGLET)
    dslashParam.threads = in->Volume();
  else //twist doublet    
    dslashParam.threads = in->Volume() / 2;

#if (defined GPU_TWISTED_MASS_DIRAC) || (defined GPU_NDEG_TWISTED_MASS_DIRAC)
  Tunable *twistGamma5 = 0;

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
    twistGamma5 = new TwistGamma5Cuda<double2>(out, in, kappa, mu, epsilon, dagger, twist);
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    twistGamma5 = new TwistGamma5Cuda<float4>(out, in, kappa, mu, epsilon, dagger, twist);
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    twistGamma5 = new TwistGamma5Cuda<short4>(out, in, kappa, mu, epsilon, dagger, twist);
  }

  twistGamma5->apply(streams[Nstream-1]);
  checkCudaError();

  delete twistGamma5;
#else
  errorQuda("Twisted mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
}

#if defined(GPU_TWISTED_CLOVER_DIRAC)
#include "dslash_core/tmc_gamma_core.h"
#endif

template <typename cFloat, typename sFloat>
class TwistCloverGamma5Cuda : public Tunable {
  private:
    const cFloat *clover;
    const float *cNorm;
    const cFloat *cloverInv;
    const float *cNrm2;
    QudaTwistGamma5Type twist;
    cudaColorSpinorField *out;
    const cudaColorSpinorField *in;

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return in->X(0) * in->X(1) * in->X(2) * in->X(3); }
    char *saveOut, *saveOutNorm;
    char aux_string[TuneKey::aux_n];

  public:
    TwistCloverGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in,
        double kappa, double mu, double epsilon, const int dagger, QudaTwistGamma5Type tw,
			  cFloat *clov, const float *cN, cFloat *clovInv, const float *cN2, int cl_stride) :
      clover(clov), cNorm(cN), cloverInv(clovInv), cNrm2(cN2), twist(tw), out(out), in(in)
    {
      bindSpinorTex<sFloat>(in);
      dslashParam.out = (void*)out->V();
      dslashParam.outNorm = (float*)out->Norm();
      dslashParam.in = (void*)in->V();
      dslashParam.inNorm = (float*)in->Norm();
      dslashParam.clover = (void*)clov;
      dslashParam.cloverNorm = (float*)cN;
      dslashParam.clover = (void*)clovInv;
      dslashParam.cloverNorm = (float*)cN2;
      dslashParam.sp_stride = in->Stride();
#ifdef GPU_TWISTED_CLOVER_DIRAC
      dslashParam.cl_stride = cl_stride;
      dslashParam.fl_stride = in->VolumeCB();
#endif

      if(in->TwistFlavor() == QUDA_TWIST_SINGLET) {
	setTwistParam(dslashParam.a, dslashParam.b, kappa, mu, dagger, tw);
      } else {//twist doublet
	errorQuda("ERROR: Non-degenerated twisted-mass not supported in this regularization\n");
      }
      dslashParam.a_f = dslashParam.a;
      dslashParam.b_f = dslashParam.b;

      strcpy(aux_string,in->AuxString());
      strcat(aux_string, twist == QUDA_TWIST_GAMMA5_DIRECT ? ",direct" : ",inverse");
    }

    virtual ~TwistCloverGamma5Cuda() { unbindSpinorTex<sFloat>(in); }

    TuneKey tuneKey() const {
      return TuneKey(in->VolString(), typeid(*this).name(), aux_string);
    }  

    void apply(const cudaStream_t &stream)
    {
#if defined(GPU_TWISTED_CLOVER_DIRAC)
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      dim3 gridDim( (dslashParam.threads+tp.block.x-1) / tp.block.x, 1, 1);
      if(in->TwistFlavor() == QUDA_TWIST_SINGLET) {	//Idea for the kernel, two spinor inputs (IN and clover applied IN), on output (Clover applied IN + ig5IN)
        if (twist == QUDA_TWIST_GAMMA5_DIRECT)
          twistCloverGamma5Kernel<sFloat><<<gridDim, tp.block, tp.shared_bytes, stream>>>(dslashParam);
        else if (twist == QUDA_TWIST_GAMMA5_INVERSE)
          twistCloverGamma5InvKernel<sFloat><<<gridDim, tp.block, tp.shared_bytes, stream>>>(dslashParam);
      } else {
        errorQuda("ERROR: Non-degenerated twisted-mass not supported in this regularization\n");
      }
#endif
    }

    void preTune() {
      saveOut = new char[out->Bytes()];
      cudaMemcpy(saveOut, out->V(), out->Bytes(), cudaMemcpyDeviceToHost);
      if (typeid(sFloat) == typeid(short4)) {
        saveOutNorm = new char[out->NormBytes()];
        cudaMemcpy(saveOutNorm, out->Norm(), out->NormBytes(), cudaMemcpyDeviceToHost);
      }
    }

    void postTune() {
      cudaMemcpy(out->V(), saveOut, out->Bytes(), cudaMemcpyHostToDevice);
      delete[] saveOut;
      if (typeid(sFloat) == typeid(short4)) {
        cudaMemcpy(out->Norm(), saveOutNorm, out->NormBytes(), cudaMemcpyHostToDevice);
        delete[] saveOutNorm;
      }
    }


    long long flops() const { return 24ll * in->VolumeCB(); }	//TODO FIX THIS NUMBER!!!
    long long bytes() const { return in->Bytes() + in->NormBytes() + out->Bytes() + out->NormBytes(); }
};

void twistCloverGamma5Cuda(cudaColorSpinorField *out, const cudaColorSpinorField *in, const int dagger, const double &kappa, const double &mu,
    const double &epsilon, const QudaTwistGamma5Type twist, const FullClover *clov, const FullClover *clovInv, const int parity)
{
  if(in->TwistFlavor() == QUDA_TWIST_SINGLET)
    dslashParam.threads = in->Volume();
  else //twist doublet    
    errorQuda("Twisted doublet not supported in twisted clover dslash");

#ifdef GPU_TWISTED_CLOVER_DIRAC
  Tunable *tmClovGamma5 = 0;

  void *clover=0, *cNorm=0, *cloverInv=0, *cNorm2=0;
  QudaPrecision clover_prec = bindTwistedCloverTex(*clov, *clovInv, parity, &clover, &cNorm, &cloverInv, &cNorm2);

  if (in->Precision() != clover_prec)
    errorQuda("ERROR: Clover precision and spinor precision do not match\n");

#ifndef DYNAMIC_CLOVER
  if (clov->stride != clovInv->stride) 
    errorQuda("clover and cloverInv must have matching strides (%d != %d)", clov->stride, clovInv->stride);
#endif
    

  if (in->Precision() == QUDA_DOUBLE_PRECISION) {
    tmClovGamma5 = new TwistCloverGamma5Cuda<double2,double2>
      (out, in, kappa, mu, epsilon, dagger, twist, (double2 *) clover, (float *) cNorm, (double2 *) cloverInv, (float *) cNorm2, clov->stride);
  } else if (in->Precision() == QUDA_SINGLE_PRECISION) {
    tmClovGamma5 = new TwistCloverGamma5Cuda<float4,float4>
      (out, in, kappa, mu, epsilon, dagger, twist, (float4 *) clover, (float *) cNorm, (float4 *) cloverInv, (float *) cNorm2, clov->stride);
  } else if (in->Precision() == QUDA_HALF_PRECISION) {
    tmClovGamma5 = new TwistCloverGamma5Cuda<short4,short4>
      (out, in, kappa, mu, epsilon, dagger, twist, (short4 *) clover, (float *) cNorm, (short4 *) cloverInv, (float *) cNorm2, clov->stride);
  }

  tmClovGamma5->apply(streams[Nstream-1]);
  checkCudaError();

  delete tmClovGamma5;
  unbindTwistedCloverTex(*clov);
#else
  errorQuda("Twisted clover dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
}

} // namespace quda

#include "contract.cu"
