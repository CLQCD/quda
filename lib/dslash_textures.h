#include <typeinfo>

// Use this macro for texture fetching for supporting either texture objects of texture references
#ifdef USE_TEXTURE_OBJECTS
#define TEX1DFETCH(type, tex, idx) tex1Dfetch<type>((tex), idx)
#else
#define TEX1DFETCH(type, tex, idx) tex1Dfetch((tex), idx)
#endif

template<typename Tex>
static __inline__ __device__ double fetch_double(Tex t, int i)
{
  int2 v = TEX1DFETCH(int2, t, i);
  return __hiloint2double(v.y, v.x);
}

template <typename Tex>
static __inline__ __device__ double2 fetch_double2(Tex t, int i)
{
  int4 v = TEX1DFETCH(int4, t, i);
  return make_double2(__hiloint2double(v.y, v.x), __hiloint2double(v.w, v.z));
}

#ifndef USE_TEXTURE_OBJECTS
// Double precision gauge field
texture<int4, 1> gauge0TexDouble2;
texture<int4, 1> gauge1TexDouble2;

// Single precision gauge field
texture<float4, 1, cudaReadModeElementType> gauge0TexSingle4;
texture<float4, 1, cudaReadModeElementType> gauge1TexSingle4;
texture<float2, 1, cudaReadModeElementType> gauge0TexSingle2;
texture<float2, 1, cudaReadModeElementType> gauge1TexSingle2;

// Half precision gauge field
texture<short4, 1, cudaReadModeNormalizedFloat> gauge0TexHalf4;
texture<short4, 1, cudaReadModeNormalizedFloat> gauge1TexHalf4;
texture<short2, 1, cudaReadModeNormalizedFloat> gauge0TexHalf2;
texture<short2, 1, cudaReadModeNormalizedFloat> gauge1TexHalf2;

// Quarter precision gauge field
texture<char4, 1, cudaReadModeNormalizedFloat> gauge0TexQuarter4;
texture<char4, 1, cudaReadModeNormalizedFloat> gauge1TexQuarter4;
texture<char2, 1, cudaReadModeNormalizedFloat> gauge0TexQuarter2;
texture<char2, 1, cudaReadModeNormalizedFloat> gauge1TexQuarter2;

texture<int4, 1> longGauge0TexDouble;
texture<int4, 1> longGauge1TexDouble;
texture<int2, 1> longPhase0TexDouble;
texture<int2, 1> longPhase1TexDouble;

texture<float4, 1, cudaReadModeElementType> longGauge0TexSingle;
texture<float4, 1, cudaReadModeElementType> longGauge1TexSingle;
texture<float2, 1, cudaReadModeElementType> longGauge0TexSingle_norecon;
texture<float2, 1, cudaReadModeElementType> longGauge1TexSingle_norecon;
texture<float, 1, cudaReadModeElementType> longPhase0TexSingle;
texture<float, 1, cudaReadModeElementType> longPhase1TexSingle;


texture<short4, 1, cudaReadModeNormalizedFloat> longGauge0TexHalf;
texture<short4, 1, cudaReadModeNormalizedFloat> longGauge1TexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> longGauge0TexHalf_norecon;
texture<short2, 1, cudaReadModeNormalizedFloat> longGauge1TexHalf_norecon;
texture<short, 1, cudaReadModeNormalizedFloat> longPhase0TexHalf;
texture<short, 1, cudaReadModeNormalizedFloat> longPhase1TexHalf;

texture<char4, 1, cudaReadModeNormalizedFloat> longGauge0TexQuarter;
texture<char4, 1, cudaReadModeNormalizedFloat> longGauge1TexQuarter;
texture<char2, 1, cudaReadModeNormalizedFloat> longGauge0TexQuarter_norecon;
texture<char2, 1, cudaReadModeNormalizedFloat> longGauge1TexQuarter_norecon;
texture<char, 1, cudaReadModeNormalizedFloat> longPhase0TexQuarter;
texture<char, 1, cudaReadModeNormalizedFloat> longPhase1TexQuarter;


// Double precision input spinor field
texture<int4, 1> spinorTexDouble;
texture<int4, 1> ghostSpinorTexDouble;

// Single precision input spinor field
texture<float4, 1, cudaReadModeElementType> spinorTexSingle;
texture<float2, 1, cudaReadModeElementType> spinorTexSingle2;
texture<float4, 1, cudaReadModeElementType> ghostSpinorTexSingle;
texture<float2, 1, cudaReadModeElementType> ghostSpinorTexSingle2;

// Half precision input spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> spinorTexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> spinorTexHalf2;
texture<float, 1, cudaReadModeElementType> spinorTexHalfNorm;
texture<float, 1, cudaReadModeElementType> spinorTexHalf2Norm;
texture<short4, 1, cudaReadModeNormalizedFloat> ghostSpinorTexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> ghostSpinorTexHalf2;
texture<float, 1, cudaReadModeElementType> ghostSpinorTexHalfNorm;
texture<float, 1, cudaReadModeElementType> ghostSpinorTexHalf2Norm;

// Quarter precision input spinor field
texture<char4, 1, cudaReadModeNormalizedFloat> spinorTexQuarter;
texture<char2, 1, cudaReadModeNormalizedFloat> spinorTexQuarter2;
texture<float, 1, cudaReadModeElementType> spinorTexQuarterNorm;
texture<float, 1, cudaReadModeElementType> spinorTexQuarter2Norm;
texture<char4, 1, cudaReadModeNormalizedFloat> ghostSpinorTexQuarter;
texture<char2, 1, cudaReadModeNormalizedFloat> ghostSpinorTexQuarter2;
texture<float, 1, cudaReadModeElementType> ghostSpinorTexQuarterNorm;
texture<float, 1, cudaReadModeElementType> ghostSpinorTexQuarter2Norm;

// Double precision accumulate spinor field
texture<int4, 1> accumTexDouble;

// Single precision accumulate spinor field
texture<float4, 1, cudaReadModeElementType> accumTexSingle;
texture<float2, 1, cudaReadModeElementType> accumTexSingle2;

// Half precision accumulate spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> accumTexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> accumTexHalf2;
texture<float, 1, cudaReadModeElementType> accumTexHalfNorm;
texture<float, 1, cudaReadModeElementType> accumTexHalf2Norm;

// Quarter precision accumulate spinor field
texture<char4, 1, cudaReadModeNormalizedFloat> accumTexQuarter;
texture<char2, 1, cudaReadModeNormalizedFloat> accumTexQuarter2;
texture<float, 1, cudaReadModeElementType> accumTexHalfNorm;
texture<float, 1, cudaReadModeElementType> accumTexHalf2Norm;

// Double precision intermediate spinor field (used by exterior Dslash kernels)
texture<int4, 1> interTexDouble;

// Single precision intermediate spinor field
texture<float4, 1, cudaReadModeElementType> interTexSingle;
texture<float2, 1, cudaReadModeElementType> interTexSingle2;

// Half precision intermediate spinor field
texture<short4, 1, cudaReadModeNormalizedFloat> interTexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> interTexHalf2;
texture<float, 1, cudaReadModeElementType> interTexHalfNorm;
texture<float, 1, cudaReadModeElementType> interTexHalf2Norm;

// Quarter precision intermediate spinor field
texture<char4, 1, cudaReadModeNormalizedFloat> interTexChar;
texture<char2, 1, cudaReadModeNormalizedFloat> interTexChar2;
texture<float, 1, cudaReadModeElementType> interTexHalfNorm;
texture<float, 1, cudaReadModeElementType> interTexHalf2Norm;
#endif // not defined USE_TEXTURE_OBJECTS

// FIXME update the below textures for texture objects

// fatGauge textures are still used by llfat so we need to define
texture<int4, 1> fatGauge0TexDouble;
texture<int4, 1> fatGauge1TexDouble;
texture<float2, 1, cudaReadModeElementType> fatGauge0TexSingle;
texture<float2, 1, cudaReadModeElementType> fatGauge1TexSingle;
texture<short2, 1, cudaReadModeNormalizedFloat> fatGauge0TexHalf;
texture<short2, 1, cudaReadModeNormalizedFloat> fatGauge1TexHalf;
texture<char2, 1, cudaReadModeNormalizedFloat> fatGauge0TexQuarter;
texture<char2, 1, cudaReadModeNormalizedFloat> fatGauge1TexQuarter;

//Double precision for site link
texture<int4, 1> siteLink0TexDouble;
texture<int4, 1> siteLink1TexDouble;

//Single precision for site link
texture<float2, 1, cudaReadModeElementType> siteLink0TexSingle;
texture<float2, 1, cudaReadModeElementType> siteLink1TexSingle;

texture<float4, 1, cudaReadModeElementType> siteLink0TexSingle_recon;
texture<float4, 1, cudaReadModeElementType> siteLink1TexSingle_recon;

texture<float2, 1, cudaReadModeElementType> siteLink0TexSingle_norecon;
texture<float2, 1, cudaReadModeElementType> siteLink1TexSingle_norecon;


texture<int4, 1> muLink0TexDouble;
texture<int4, 1> muLink1TexDouble;
// Single precision mulink field
texture<float2, 1, cudaReadModeElementType> muLink0TexSingle;
texture<float2, 1, cudaReadModeElementType> muLink1TexSingle;

template<typename T>
void bindGaugeTex(const cudaGaugeField &gauge, const int oddBit, T &dslashParam)
{
  if(oddBit) {
    dslashParam.gauge0 = const_cast<void*>(gauge.Odd_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Even_p());
  } else {
    dslashParam.gauge0 = const_cast<void*>(gauge.Even_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Odd_p());
  }

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.gauge0Tex = oddBit ? gauge.OddTex() : gauge.EvenTex();
  dslashParam.gauge1Tex = oddBit ? gauge.EvenTex() : gauge.OddTex();
#else
  if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    if (gauge.Precision() == QUDA_DOUBLE_PRECISION) {
      cudaBindTexture(0, gauge0TexDouble2, dslashParam.gauge0, gauge.Bytes()/2);
      cudaBindTexture(0, gauge1TexDouble2, dslashParam.gauge1, gauge.Bytes()/2);
    } else if (gauge.Precision() == QUDA_SINGLE_PRECISION) {
      cudaBindTexture(0, gauge0TexSingle2, dslashParam.gauge0, gauge.Bytes()/2);
      cudaBindTexture(0, gauge1TexSingle2, dslashParam.gauge1, gauge.Bytes()/2);
    } else if (gauge.Precision == QUDA_HALF_PRECISION) {
      cudaBindTexture(0, gauge0TexHalf2, dslashParam.gauge0, gauge.Bytes()/2);
      cudaBindTexture(0, gauge1TexHalf2, dslashParam.gauge1, gauge.Bytes()/2);
    } else if (gauge.Precision() == QUDA_QUARTER_PRECISION) {
      cudaBindTexture(0, gauge0TexQuarter2, dslashParam.gauge0, gauge.Bytes()/2); 
      cudaBindTexture(0, gauge1TexQuarter2, dslashParam.gauge1, gauge.Bytes()/2);
    } else {
      errorQuda("gauge precision %d is not supported", gauge.Precision());
    }
  } else {
    if (gauge.Precision() == QUDA_DOUBLE_PRECISION) {
      cudaBindTexture(0, gauge0TexDouble2, dslashParam.gauge0, gauge.Bytes()/2);
      cudaBindTexture(0, gauge1TexDouble2, dslashParam.gauge1, gauge.Bytes()/2);
    } else if (gauge.Precision() == QUDA_SINGLE_PRECISION) {
      cudaBindTexture(0, gauge0TexSingle4, dslashParam.gauge0, gauge.Bytes()/2);
      cudaBindTexture(0, gauge1TexSingle4, dslashParam.gauge1, gauge.Bytes()/2);
    } else if (gauge.Precision() == QUDA_HALF_PRECISION) {
      cudaBindTexture(0, gauge0TexHalf4, dslashParam.gauge0, gauge.Bytes()/2);
      cudaBindTexture(0, gauge1TexHalf4, dslashParam.gauge1, gauge.Bytes()/2);
    } else if (gauge.Precision() == QUDA_QUARTER_PRECISION) {
      cudaBindTexture(0, gauge0TexQuarter4, dslashParam.gauge0, gauge.Bytes()/2); 
      cudaBindTexture(0, gauge1TexQuarter4, dslashParam.gauge1, gauge.Bytes()/2);
    } else {
      errorQuda("gauge precision %d is not supported", gauge.Precision());
    }
  }
#endif // USE_TEXTURE_OBJECTS

}

void unbindGaugeTex(const cudaGaugeField &gauge)
{
#if (!defined USE_TEXTURE_OBJECTS)
  if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
    if (gauge.Precision() == QUDA_DOUBLE_PRECISION) {
      cudaUnbindTexture(gauge0TexDouble2);
      cudaUnbindTexture(gauge1TexDouble2);
    } else if (gauge.Precision() == QUDA_SINGLE_PRECISION) {
      cudaUnbindTexture(gauge0TexSingle2);
      cudaUnbindTexture(gauge1TexSingle2);
    } else if (gauge.Precision() == QUDA_HALF_PRECISION) {
      cudaUnbindTexture(gauge0TexHalf2);
      cudaUnbindTexture(gauge1TexHalf2);
    } else if (gauge.Precision() == QUDA_QUARTER_PRECISION) {
      cudaUnbindTexture(gauge0TexQuarter2); 
      cudaUnbindTexture(gauge1TexQuarter2);
    } else {
      errorQuda("gauge precision %d is not supported", gauge.Precision());
    }
  } else {
    if (gauge.Precision() == QUDA_DOUBLE_PRECISION) {
      cudaUnbindTexture(gauge0TexDouble2);
      cudaUnbindTexture(gauge1TexDouble2);
    } else if (gauge.Precision() == QUDA_SINGLE_PRECISION) {
      cudaUnbindTexture(gauge0TexSingle4);
      cudaUnbindTexture(gauge1TexSingle4);
    } else if (gauge.Precision() == QUDA_HALF_PRECISION) {
      cudaUnbindTexture(gauge0TexHalf4);
      cudaUnbindTexture(gauge0TexHalf4);
    } else if (gauge.Precision() == QUDA_QUARTER_PRECISION) {
      cudaUnbindTexture(gauge0TexQuarter4); 
      cudaUnbindTexture(gauge1TexQuarter4);
    } else {
      errorQuda("gauge precision %d is not supported", gauge.Precision());
    }
  }
#endif
}

template <typename T>
void bindFatGaugeTex(const cudaGaugeField &gauge, const int oddBit, T &dslashParam)
{
  if(oddBit) {
    dslashParam.gauge0 = const_cast<void*>(gauge.Odd_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Even_p());
  } else {
    dslashParam.gauge0 = const_cast<void*>(gauge.Even_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Odd_p());
  }

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.gauge0Tex = oddBit ? gauge.OddTex() : gauge.EvenTex();
  dslashParam.gauge1Tex = oddBit ? gauge.EvenTex() : gauge.OddTex();
#else
  if (gauge.Precision() == QUDA_DOUBLE_PRECISION) {
    cudaBindTexture(0, fatGauge0TexDouble, dslashParam.gauge0, gauge.Bytes()/2);
    cudaBindTexture(0, fatGauge1TexDouble, dslashParam.gauge1, gauge.Bytes()/2);
  } else if (gauge.Precision() == QUDA_SINGLE_PRECISION) {
    cudaBindTexture(0, fatGauge0TexSingle, dslashParam.gauge0, gauge.Bytes()/2);
    cudaBindTexture(0, fatGauge1TexSingle, dslashParam.gauge1, gauge.Bytes()/2);
  } else if (gauge.Precision() == QUDA_HALF_PRECISION) {
    cudaBindTexture(0, fatGauge0TexHalf, dslashParam.gauge0, gauge.Bytes()/2);
    cudaBindTexture(0, fatGauge1TexHalf, dslashParam.gauge1, gauge.Bytes()/2);
  } else if (gauge.Precision() == QUDA_QUARTER_PRECISION) {
    cudaBindTexture(0, fatGauge0TexQuarter, dslashParam.gauge0, gauge.Bytes()/2); 
    cudaBindTexture(0, fatGauge1TexQuarter, dslashParam.gauge1, gauge.Bytes()/2);
  } else {
    errorQuda("gauge precision %d is not supported", gauge.Precision());
  }
#endif // USE_TEXTURE_OBJECTS

}

void unbindFatGaugeTex(const cudaGaugeField &gauge)
{
#if (!defined USE_TEXTURE_OBJECTS)
  if (gauge.Precision() == QUDA_DOUBLE_PRECISION) {
    cudaUnbindTexture(fatGauge0TexDouble);
    cudaUnbindTexture(fatGauge1TexDouble);
  } else if (gauge.Precision() == QUDA_SINGLE_PRECISION) {
    cudaUnbindTexture(fatGauge0TexSingle);
    cudaUnbindTexture(fatGauge1TexSingle);
  } else if (gauge.Precision() == QUDA_HALF_PRECISION) {
    cudaUnbindTexture(fatGauge0TexHalf);
    cudaUnbindTexture(fatGauge1TexHalf);
  } else if (gauge.Precision() == QUDA_QUARTER_PRECISION) {
    cudaUnbindTexture(fatGauge0TexQuarter);
    cudaUnbindTexture(fatGauge1TexQuarter);
  } else {
    errorQuda("gauge precision is not supported");
  }
#endif
}

template <typename T>
void bindLongGaugeTex(const cudaGaugeField &gauge, const int oddBit, T &dslashParam)
{
  if (oddBit) {
    dslashParam.gauge0 = const_cast<void*>(gauge.Odd_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Even_p());
  } else {
    dslashParam.gauge0 = const_cast<void*>(gauge.Even_p());
    dslashParam.gauge1 = const_cast<void*>(gauge.Odd_p());
  }

  dslashParam.longPhase0 = static_cast<char*>(dslashParam.longGauge0) + gauge.PhaseOffset();
  dslashParam.longPhase1 = static_cast<char*>(dslashParam.longGauge1) + gauge.PhaseOffset();


#ifdef USE_TEXTURE_OBJECTS
  dslashParam.longGauge0Tex = oddBit ? gauge.OddTex() : gauge.EvenTex();
  dslashParam.longGauge1Tex = oddBit ? gauge.EvenTex() : gauge.OddTex();

  if(gauge.Reconstruct() == QUDA_RECONSTRUCT_13 || gauge.Reconstruct() == QUDA_RECONSTRUCT_9){
    dslashParam.longPhase0Tex = oddBit ? gauge.OddPhaseTex() : gauge.EvenPhaseTex();
    dslashParam.longPhase1Tex = oddBit ? gauge.EvenPhaseTex() : gauge.OddPhaseTex();
  }
#else
  if (gauge.Precision() == QUDA_DOUBLE_PRECISION) {
    cudaBindTexture(0, longGauge0TexDouble, dslashParam.gauge0, gauge.Bytes()/2);
    cudaBindTexture(0, longGauge1TexDouble, dslashParam.gauge1, gauge.Bytes()/2);
    if(gauge.Reconstruct() == QUDA_RECONSTRUCT_13 || gauge.Reconstruct() == QUDA_RECONSTRUCT_9){
      cudaBindTexture(0, longPhase0TexDouble, (char*)(dslashParam.gauge0) + gauge.PhaseOffset(), gauge.PhaseBytes()/2);
      cudaBindTexture(0, longPhase1TexDouble, (char*)(dslashParam.gauge1) + gauge.PhaseOffset(), gauge.PhaseBytes()/2);
    }
  } else if (gauge.Precision() == QUDA_SINGLE_PRECISION) {
    if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) { //18 reconstruct
      cudaBindTexture(0, longGauge0TexSingle_norecon, dslashParam.gauge0, gauge.Bytes()/2);
      cudaBindTexture(0, longGauge1TexSingle_norecon, dslashParam.gauge1, gauge.Bytes()/2);
    } else {
      cudaBindTexture(0, longGauge0TexSingle, dslashParam.gauge0, gauge.Bytes()/2);
      cudaBindTexture(0, longGauge1TexSingle, dslashParam.gauge1, gauge.Bytes()/2);
      if(gauge.Reconstruct() == QUDA_RECONSTRUCT_13 || gauge.Reconstruct() == QUDA_RECONSTRUCT_9){
        cudaBindTexture(0, longPhase0TexSingle, (char*)(dslashParam.gauge0) + gauge.PhaseOffset(), gauge.PhaseBytes()/2);
        cudaBindTexture(0, longPhase1TexSingle, (char*)(dslashParam.gauge1) + gauge.PhaseOffset(), gauge.PhaseBytes()/2);
      }
    }
  } else if (gauge.Precision() == QUDA_HALF_PRECISION) {
    if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) { //18 reconstruct
      cudaBindTexture(0, longGauge0TexHalf_norecon, dslashParam.gauge0, gauge.Bytes()/2);
      cudaBindTexture(0, longGauge1TexHalf_norecon, dslashParam.gauge1, gauge.Bytes()/2);
    } else {
      cudaBindTexture(0, longGauge0TexHalf, dslashParam.gauge0, gauge.Bytes()/2);
      cudaBindTexture(0, longGauge1TexHalf, dslashParam.gauge1, gauge.Bytes()/2);
      if(gauge.Reconstruct() == QUDA_RECONSTRUCT_13 || gauge.Reconstruct() == QUDA_RECONSTRUCT_9){
        cudaBindTexture(0, longPhase0TexHalf, (char*)(dslashParam.gauge0) + gauge.PhaseOffset(), gauge.PhaseBytes()/2);
	cudaBindTexture(0, longPhase1TexHalf, (char*)(dslashParam.gauge1) + gauge.PhaseOffset(), gauge.PhaseBytes()/2);
      }
    }
  } else if (gauge.Precision() == QUDA_QUARTER_PRECISION) {
    if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) { //18 reconstruct
      cudaBindTexture(0, longGauge0TexQuarter_norecon, *gauge0, gauge.Bytes()/2); 
      cudaBindTexture(0, longGauge1TexQuarter_norecon, *gauge1, gauge.Bytes()/2);  
    } else {
      cudaBindTexture(0, longGauge0TexQuarter, *gauge0, gauge.Bytes()/2); 
      cudaBindTexture(0, longGauge1TexQuarter, *gauge1, gauge.Bytes()/2);
      if(gauge.Reconstruct() == QUDA_RECONSTRUCT_13 || gauge.Reconstruct() == QUDA_RECONSTRUCT_9){
        cudaBindTexture(0, longPhase0TexQuarter, (char*)(*gauge0) + gauge.PhaseOffset(), gauge.PhaseBytes()/2);
  cudaBindTexture(0, longPhase1TexQuarter, (char*)(*gauge1) + gauge.PhaseOffset(), gauge.PhaseBytes()/2);
      }
    }
  } else {
    errorQuda("gauge precision not supported");
  }
#endif // USE_TEXTURE_OBJECTS
}

void unbindLongGaugeTex(const cudaGaugeField &gauge)
{
#if (!defined USE_TEXTURE_OBJECTS)
  if (gauge.Precision() == QUDA_DOUBLE_PRECISION) {
    cudaUnbindTexture(longGauge0TexDouble);
    cudaUnbindTexture(longGauge1TexDouble);
    if(gauge.Reconstruct() == QUDA_RECONSTRUCT_13 || gauge.Reconstruct() == QUDA_RECONSTRUCT_9){
      cudaUnbindTexture(longPhase0TexDouble);
      cudaUnbindTexture(longPhase1TexDouble);
    }
  } else if (gauge.Precision() == QUDA_SINGLE_PRECISION) {
    if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) { //18 reconstruct
      cudaUnbindTexture(longGauge0TexSingle_norecon);
      cudaUnbindTexture(longGauge1TexSingle_norecon);
    } else {
      cudaUnbindTexture(longGauge0TexSingle);
      cudaUnbindTexture(longGauge1TexSingle);
      if(gauge.Reconstruct() == QUDA_RECONSTRUCT_13 || gauge.Reconstruct() == QUDA_RECONSTRUCT_9){
        cudaUnbindTexture(longPhase0TexSingle);
        cudaUnbindTexture(longPhase1TexSingle);
      }
    }
  } else if (gauge.Precision() == QUDA_HALF_PRECISION) { // half precision
    if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) { //18 reconstruct
      cudaUnbindTexture(longGauge0TexHalf_norecon);
      cudaUnbindTexture(longGauge1TexHalf_norecon);
    } else {
      cudaUnbindTexture(longGauge0TexHalf);
      cudaUnbindTexture(longGauge1TexHalf);
      if(gauge.Reconstruct() == QUDA_RECONSTRUCT_13 || gauge.Reconstruct() ==  QUDA_RECONSTRUCT_9){
        cudaUnbindTexture(longPhase0TexHalf);
        cudaUnbindTexture(longPhase1TexHalf);
      }
    }
  } else if (gauge.Precision() == QUDA_QUARTER_PRECISION) { 
    if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) { //18 reconstruct
      cudaUnbindTexture(longGauge0TexQuarter_norecon);
      cudaUnbindTexture(longGauge1TexQuarter_norecon);
    } else {
      cudaUnbindTexture(longGauge0TexQuarter);
      cudaUnbindTexture(longGauge1TexQuarter);
      if(gauge.Reconstruct() == QUDA_RECONSTRUCT_13 || gauge.Reconstruct() ==  QUDA_RECONSTRUCT_9){
        cudaUnbindTexture(longPhase0TexQuarter);
        cudaUnbindTexture(longPhase1TexQuarter);
      }
    }
  } else {
    errorQuda("gauge precision not supported");
  }
#endif
}


template <typename spinorFloat>
int bindSpinorTex(const cudaColorSpinorField *in, const cudaColorSpinorField *out=0,
		  const cudaColorSpinorField *x=0) {
  int size = (sizeof(((spinorFloat*)0)->x) < sizeof(float)) ? sizeof(float) :
    sizeof(((spinorFloat*)0)->x);

#ifndef USE_TEXTURE_OBJECTS
  if (typeid(spinorFloat) == typeid(double2)) {
    cudaBindTexture(0, spinorTexDouble, in->V(), in->Bytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexDouble, in->Ghost2(), in->GhostBytes());
    if (out) cudaBindTexture(0, interTexDouble, out->V(), in->Bytes());
    if (x) cudaBindTexture(0, accumTexDouble, x->V(), in->Bytes());
  } else if (typeid(spinorFloat) == typeid(float4)) {
    cudaBindTexture(0, spinorTexSingle, in->V(), in->Bytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexSingle, in->Ghost2(), in->GhostBytes());
    if (out) cudaBindTexture(0, interTexSingle, out->V(), in->Bytes());
    if (x) cudaBindTexture(0, accumTexSingle, x->V(), in->Bytes());
  } else if  (typeid(spinorFloat) == typeid(float2)) {
    cudaBindTexture(0, spinorTexSingle2, in->V(), in->Bytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexSingle2, in->Ghost2(), in->GhostBytes());
    if (out) cudaBindTexture(0, interTexSingle2, out->V(), in->Bytes());
    if (x) cudaBindTexture(0, accumTexSingle2, x->V(), in->Bytes());
  } else if (typeid(spinorFloat) == typeid(short4)) {
    cudaBindTexture(0, spinorTexHalf, in->V(), in->Bytes());
    cudaBindTexture(0, spinorTexHalfNorm, in->Norm(), in->NormBytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexHalf, in->Ghost2(), in->GhostBytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexHalfNorm, in->Ghost2(), in->GhostBytes());
    if (out) cudaBindTexture(0, interTexHalf, out->V(), in->Bytes());
    if (out) cudaBindTexture(0, interTexHalfNorm, out->Norm(), in->NormBytes());
    if (x) cudaBindTexture(0, accumTexHalf, x->V(), in->Bytes());
    if (x) cudaBindTexture(0, accumTexHalfNorm, x->Norm(), in->NormBytes());
  } else if (typeid(spinorFloat) == typeid(short2)) {
    cudaBindTexture(0, spinorTexHalf2, in->V(), in->Bytes());
    cudaBindTexture(0, spinorTexHalf2Norm, in->Norm(), in->NormBytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexHalf2, in->Ghost2(), in->GhostBytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexHalf2Norm, in->Ghost2(), in->GhostBytes());
    if (out) cudaBindTexture(0, interTexHalf2, out->V(), in->Bytes());
    if (out) cudaBindTexture(0, interTexHalf2Norm, out->Norm(), in->NormBytes());
    if (x) cudaBindTexture(0, accumTexHalf2, x->V(), in->Bytes());
    if (x) cudaBindTexture(0, accumTexHalf2Norm, x->Norm(), in->NormBytes());
  } else if (typeid(spinorFloat) == typeid(char4)) {
    cudaBindTexture(0, spinorTexQuarter, in->V(), in->Bytes());
    cudaBindTexture(0, spinorTexQuarterNorm, in->Norm(), in->NormBytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexQuarter, in->Ghost2(), in->GhostBytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexQuarterNorm, in->Ghost2(), in->GhostBytes());
    if (out) cudaBindTexture(0, interTexQuarter, out->V(), in->Bytes());
    if (out) cudaBindTexture(0, interTexQuarterNorm, out->Norm(), in->NormBytes());
    if (x) cudaBindTexture(0, accumTexQuarter, x->V(), in->Bytes());
    if (x) cudaBindTexture(0, accumTexQuarterNorm, x->Norm(), in->NormBytes());
  } else if (typeid(spinorFloat) == typeid(char2)) {
    cudaBindTexture(0, spinorTexQuarter2, in->V(), in->Bytes());
    cudaBindTexture(0, spinorTexQuarter2Norm, in->Norm(), in->NormBytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexQuarter2, in->Ghost2(), in->GhostBytes());
    if (in->GhostBytes()) cudaBindTexture(0, ghostSpinorTexQuarter2Norm, in->Ghost2(), in->GhostBytes());
    if (out) cudaBindTexture(0, interTexQuarter, out->V(), in->Bytes());
    if (out) cudaBindTexture(0, interTexQuarter2Norm, out->Norm(), in->NormBytes());
    if (x) cudaBindTexture(0, accumTexQuarter2, x->V(), in->Bytes());
    if (x) cudaBindTexture(0, accumTexQuarter2Norm, x->Norm(), in->NormBytes());
  }else {
    errorQuda("Unsupported precision and short vector type");
  }
#endif // !USE_TEXTURE_OBJECTS

  return size;
}

template <typename spinorFloat>
void unbindSpinorTex(const cudaColorSpinorField *in, const cudaColorSpinorField *out=0,
		     const cudaColorSpinorField *x=0) {
#ifndef USE_TEXTURE_OBJECTS
  if (typeid(spinorFloat) == typeid(double2)) {
    cudaUnbindTexture(spinorTexDouble);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexDouble);
    if (out) cudaUnbindTexture(interTexDouble);
    if (x) cudaUnbindTexture(accumTexDouble);
  } else if (typeid(spinorFloat) == typeid(float4)) {
    cudaUnbindTexture(spinorTexSingle);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexSingle);
    if (out) cudaUnbindTexture(interTexSingle);
    if (x) cudaUnbindTexture(accumTexSingle);
  } else if  (typeid(spinorFloat) == typeid(float2)) {
    cudaUnbindTexture(spinorTexSingle2);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexSingle2);
    if (out) cudaUnbindTexture(interTexSingle2);
    if (x) cudaUnbindTexture(accumTexSingle2);
  } else if (typeid(spinorFloat) == typeid(short4)) {
    cudaUnbindTexture(spinorTexHalf);
    cudaUnbindTexture(spinorTexHalfNorm);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexHalf);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexHalfNorm);
    if (out) cudaUnbindTexture(interTexHalf);
    if (out) cudaUnbindTexture(interTexHalfNorm);
    if (x) cudaUnbindTexture(accumTexHalf);
    if (x) cudaUnbindTexture(accumTexHalfNorm);
  } else if (typeid(spinorFloat) == typeid(short2)) {
    cudaUnbindTexture(spinorTexHalf2);
    cudaUnbindTexture(spinorTexHalf2Norm);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexHalf2);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexHalf2Norm);
    if (out) cudaUnbindTexture(interTexHalf2);
    if (out) cudaUnbindTexture(interTexHalf2Norm);
    if (x) cudaUnbindTexture(accumTexHalf2);
    if (x) cudaUnbindTexture(accumTexHalf2Norm);
  } else if (typeid(spinorFloat) == typeid(char4)) {
    cudaUnbindTexture(spinorTexQuarter); 
    cudaUnbindTexture(spinorTexQuarterNorm);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexQuarter);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexQuarterNorm);
    if (out) cudaUnbindTexture(interTexQuarter); 
    if (out) cudaUnbindTexture(interTexQuarterNorm);
    if (x) cudaUnbindTexture(accumTexQuarter); 
    if (x) cudaUnbindTexture(accumTexQuarterNorm);
  } else if (typeid(spinorFloat) == typeid(char2)) {
    cudaUnbindTexture(spinorTexQuarter2); 
    cudaUnbindTexture(spinorTexQuarter2Norm);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexQuarter2);
    if (in->GhostBytes()) cudaUnbindTexture(ghostSpinorTexQuarter2Norm);
    if (out) cudaUnbindTexture(interTexQuarter2); 
    if (out) cudaUnbindTexture(interTexQuarter2Norm);
    if (x) cudaUnbindTexture(accumTexQuarter2); 
    if (x) cudaUnbindTexture(accumTexQuarter2Norm);
  } else {
    errorQuda("Unsupported precision and short vector type");
  }
#endif // USE_TEXTURE_OBJECTS
}

// Double precision clover term
texture<int4, 1> cloverTexDouble;
texture<int4, 1> cloverInvTexDouble;

// Single precision clover term
texture<float4, 1, cudaReadModeElementType> cloverTexSingle;
texture<float4, 1, cudaReadModeElementType> cloverInvTexSingle;

// Norms for half and quarter precision clover terms.
texture<float, 1, cudaReadModeElementType> cloverTexNorm;
texture<float, 1, cudaReadModeElementType> cloverInvTexNorm;

// Half precision clover term
texture<short4, 1, cudaReadModeNormalizedFloat> cloverTexHalf;
texture<short4, 1, cudaReadModeNormalizedFloat> cloverInvTexHalf;

// Quarter precision clover term, use norms from half.
texture<char4, 1, cudaReadModeNormalizedFloat> cloverTexQuarter;
texture<char4, 1, cudaReadModeNormalizedFloat> cloverInvTexQuarter;

template <typename T>
QudaPrecision bindCloverTex(const FullClover &clover, const int oddBit, T &dslashParam)
{
  if (oddBit) {
    dslashParam.clover = clover.odd;
    dslashParam.cloverNorm = (float*)clover.oddNorm;
  } else {
    dslashParam.clover = clover.even;
    dslashParam.cloverNorm = (float*)clover.evenNorm;
  }

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.cloverTex = oddBit ? clover.OddTex() : clover.EvenTex();
  if (clover.precision == QUDA_HALF_PRECISION || clover.precision == QUDA_QUARTER_PRECISION) dslashParam.cloverNormTex = oddBit ? clover.OddNormTex() : clover.EvenNormTex();
#else
  if (clover.precision == QUDA_DOUBLE_PRECISION) {
    cudaBindTexture(0, cloverTexDouble, dslashParam.clover, clover.bytes);
  } else if (clover.precision == QUDA_SINGLE_PRECISION) {
    cudaBindTexture(0, cloverTexSingle, dslashParam.clover, clover.bytes);
  } else if (clover.precision == QUDA_HALF_PRECISION) {
    cudaBindTexture(0, cloverTexHalf, dslashParam.clover, clover.bytes); 
    cudaBindTexture(0, cloverTexNorm, dslashParam.cloverNorm, clover.norm_bytes);
  } else if (clover.precision == QUDA_QUARTER_PRECISION) {
    cudaBindTexture(0, cloverTexQuarter, dslashParam.clover, clover.bytes); 
    cudaBindTexture(0, cloverTexNorm, dslashParam.cloverNorm, clover.norm_bytes);
  } else {
    errorQuda("Unsupported precision %d", clover.precision);
  }
#endif // USE_TEXTURE_OBJECTS

  return clover.precision;
}

void unbindCloverTex(const FullClover clover)
{
#if (!defined USE_TEXTURE_OBJECTS)
  if (clover.precision == QUDA_DOUBLE_PRECISION) {
    cudaUnbindTexture(cloverTexDouble);
  } else if (clover.precision == QUDA_SINGLE_PRECISION) {
    cudaUnbindTexture(cloverTexSingle);
  } else if (clover.precision == QUDA_HALF_PRECISION) {
    cudaUnbindTexture(cloverTexHalf);
    cudaUnbindTexture(cloverTexNorm);
  } else if (clover.precision == QUDA_QUARTER_PRECISION) {
    cudaUnbindTexture(cloverTexQuarter);
    cudaUnbindTexture(cloverTexNorm);
  } else {
    errorQuda("Unsupported precision");
  }
#endif // not defined USE_TEXTURE_OBJECTS
}

template <typename T>
QudaPrecision bindTwistedCloverTex(const FullClover clover, const FullClover cloverInv, const int oddBit, T &dslashParam)
{
  if (oddBit) {
    dslashParam.clover	 = clover.odd;
    dslashParam.cloverNorm = (float*)clover.oddNorm;
#ifndef DYNAMIC_CLOVER
    dslashParam.cloverInv = cloverInv.odd;
    dslashParam.cloverInvNorm = (float*)cloverInv.oddNorm;
#endif
  } else {
    dslashParam.clover = clover.even;
    dslashParam.cloverNorm = (float*)clover.evenNorm;
#ifndef DYNAMIC_CLOVER
    dslashParam.clover = cloverInv.even;
    dslashParam.cloverInvNorm = (float*)cloverInv.evenNorm;
#endif
  }

#ifdef USE_TEXTURE_OBJECTS
  dslashParam.cloverTex = oddBit ? clover.OddTex() : clover.EvenTex();
  if (clover.precision == QUDA_HALF_PRECISION || clover.precision == QUDA_QUARTER_PRECISION) dslashParam.cloverNormTex = oddBit ? clover.OddNormTex() : clover.EvenNormTex();
#ifndef DYNAMIC_CLOVER
  dslashParam.cloverInvTex = oddBit ? cloverInv.OddTex() : cloverInv.EvenTex();
  if (cloverInv.precision == QUDA_HALF_PRECISION || clover.precision == QUDA_QUARTER_PRECISION) dslashParam.cloverInvNormTex = oddBit ? cloverInv.OddNormTex() : cloverInv.EvenNormTex();
#endif
#else
  if (clover.precision == QUDA_DOUBLE_PRECISION) {   //I assume that the clover and cloverInv fields have the same precision
    cudaBindTexture(0, cloverTexDouble, dslashParam.clover, clover.bytes);
#ifndef DYNAMIC_CLOVER
    cudaBindTexture(0, cloverInvTexDouble, dslashParam.cloverInv, cloverInv.bytes);
#endif
  } else if (clover.precision == QUDA_SINGLE_PRECISION) {
    cudaBindTexture(0, cloverTexSingle, dslashParam.clover, clover.bytes);
#ifndef DYNAMIC_CLOVER
    cudaBindTexture(0, cloverInvTexSingle, dslashParam.cloverInv, cloverInv.bytes);
#endif
  } else if (clover.precision == QUDA_HALF_PRECISION) {
    cudaBindTexture(0, cloverTexHalf, dslashParam.clover, clover.bytes);
    cudaBindTexture(0, cloverTexNorm, dslashParam.cloverNorm, clover.norm_bytes);
#ifndef DYNAMIC_CLOVER
    cudaBindTexture(0, cloverInvTexHalf, dslashParam.cloverInv, cloverInv.bytes);
    cudaBindTexture(0, cloverInvTexNorm, dslashParam.cloverInvNorm, cloverInv.norm_bytes);
#endif
  } else if (clover.precision == QUDA_QUARTER_PRECISION) {
    cudaBindTexture(0, cloverTexQuarter, dslashParam.clover, clover.bytes); 
    cudaBindTexture(0, cloverTexNorm, dslashParam.cloverNorm, clover.norm_bytes);
#ifndef DYNAMIC_CLOVER
    cudaBindTexture(0, cloverInvTexQuarter, dslashParam.cloverInv, cloverInv.bytes); 
    cudaBindTexture(0, cloverInvTexNorm, dslashParam.cloverInvNorm, cloverInv.norm_bytes);
#endif
  } else {
    errorQuda("Unsupported precision");
  }
#endif // USE_TEXTURE_OBJECTS

  return clover.precision;
}

void unbindTwistedCloverTex(const FullClover clover)  //We don't really need this function, but for the shake of completeness...
{
#if (!defined USE_TEXTURE_OBJECTS)
  if (clover.precision == QUDA_DOUBLE_PRECISION)  //Again we assume that the precision of the clover and cloverInv are the same
    {
      cudaUnbindTexture(cloverTexDouble);
#ifndef DYNAMIC_CLOVER
      cudaUnbindTexture(cloverInvTexDouble);
#endif
    }
  else if (clover.precision == QUDA_SINGLE_PRECISION)
    {
      cudaUnbindTexture(cloverTexSingle);
#ifndef DYNAMIC_CLOVER
      cudaUnbindTexture(cloverInvTexSingle);
#endif
    }
  else if (clover.precision == QUDA_HALF_PRECISION)
    {
      cudaUnbindTexture(cloverTexHalf);
      cudaUnbindTexture(cloverTexNorm);
#ifndef DYNAMIC_CLOVER
      cudaUnbindTexture(cloverInvTexHalf);
      cudaUnbindTexture(cloverInvTexNorm);
#endif
    }
  else if (clover.precision == QUDA_QUARTER_PRECISION)
    {
      cudaUnbindTexture(cloverTexQuarter);
      cudaUnbindTexture(cloverTexNorm);
#ifndef DYNAMIC_CLOVER
      cudaUnbindTexture(cloverInvTexQuarter);
      cudaUnbindTexture(cloverInvTexNorm);
#endif
    } else {
    errorQuda("Unsupported precision");
  }
#endif // not defined USE_TEXTURE_OBJECTS
}

// define some function if we're not using textures (direct access)
#if defined(DIRECT_ACCESS_LINK) || defined(DIRECT_ACCESS_WILSON_SPINOR) || \
  defined(DIRECT_ACCESS_WILSON_ACCUM) || defined(DIRECT_ACCESS_WILSON_PACK_SPINOR) || \
  defined(DIRECT_ACCESS_WILSON_INTER) || defined(DIRECT_ACCESS_WILSON_PACK_SPINOR) || \
  defined(DIRECT_ACCESS_CLOVER) || defined(DIRECT_ACCESS_PACK) ||       \
  defined(DIRECT_ACCESS_LONG_LINK) || defined(DIRECT_ACCESS_FAT_LINK)

  // Half precision
  static inline __device__ float short2float(short a) {
    return (float)a/fixedMaxValue<short>::value;
  }

  static inline __device__ short float2short(float c, float a) {
    return (short)(a*c*fixedMaxValue<short>::value);
  }

  static inline __device__ short4 float42short4(float c, float4 a) {
    return make_short4(float2short(c, a.x), float2short(c, a.y), float2short(c, a.z), float2short(c, a.w));
  }

  static inline __device__ float4 short42float4(short4 a) {
    return make_float4(short2float(a.x), short2float(a.y), short2float(a.z), short2float(a.w));
  }

  static inline __device__ float2 short22float2(short2 a) {
    return make_float2(short2float(a.x), short2float(a.y));
  }

  // Quarter precision
  static inline __device__ float char2float(char a) {
    return (float)a/fixedMaxValue<char>::value;
  }

  static inline __device__ char float2char(float c, float a) {
    return (char)(a*c*fixedMaxValue<char>::value);
  }

  static inline __device__ char4 float42char4(float c, float4 a) {
    return make_char4(float2char(c, a.x), float2char(c, a.y), float2char(c, a.z), float2char(c, a.w));
  }

  static inline __device__ float4 char42float4(char4 a) {
    return make_float4(char2float(a.x), char2float(a.y), char2float(a.z), char2float(a.w));
  }

  static inline __device__ float2 char22float2(char2 a) {
    return make_float2(char2float(a.x), char2float(a.y));
  }
#endif // DIRECT_ACCESS inclusions
