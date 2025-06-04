#include "util_quda.h"
#include <dirac_quda.h>
#include <blas_quda.h>
#include <multigrid.h>
#include <eigensolve_quda.h>
#include <gauge_field.h>

namespace quda
{

  DiracChiralOverlap::DiracChiralOverlap(const DiracParam &param) :
    Dirac(param), overlap(new DiracOverlap(param)), zero_shift(0.0)
  {
  }

  DiracChiralOverlap::DiracChiralOverlap(const DiracChiralOverlap &dirac) :
    Dirac(dirac), overlap(new DiracOverlap(*dirac.overlap)), chirality(dirac.chirality), zero_shift(dirac.zero_shift)
  {
  }

  DiracChiralOverlap::~DiracChiralOverlap() 
  { 
    if (overlap) delete overlap; 
    overlap = nullptr;
  }

  DiracChiralOverlap& DiracChiralOverlap::operator=(const DiracChiralOverlap &dirac)
  {
    if (&dirac != this) {
      Dirac::operator=(dirac);
      if (overlap) delete overlap;
      overlap = new DiracOverlap(*dirac.overlap);
      chirality = dirac.chirality;
    }
    return *this;
  }

  void DiracChiralOverlap::setOverlap(DiracOverlap *new_overlap)
  {
    if (this->overlap) delete this->overlap;
    this->overlap = new_overlap;
  }

#define flip(x) (x) = ((x) == QUDA_DAG_YES ? QUDA_DAG_NO : QUDA_DAG_YES)

  void DiracChiralOverlap::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, 
                                  QudaParity parity) const 
  { 
    errorQuda("DiracChiralOverlap::Dslash not implemented!\n"); 
  }

  void DiracChiralOverlap::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, 
                                      QudaParity parity, cvector_ref<const ColorSpinorField> &x, double scale) const 
  { 
    errorQuda("DiracChiralOverlap::DslashXpay not implemented!\n"); 
  }

  void DiracChiralOverlap::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    errorQuda("DiracChiralOverlap::M not implemented!\n");
  }

  void DiracChiralOverlap::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    printfQuda("Entering DiracChiralOverlap::MdagM\n");

    if (!overlap) {
      errorQuda("DiracChiralOverlap::MdagM: overlap pointer is null");
      return;
    }

    ColorSpinorParam fullParam(out[0]);
    fullParam.nSpin = 4;
    fullParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    fullParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField in_full(fullParam);
    ColorSpinorField out_full(fullParam);

    // in(nSpin = 2) -> in_full(nSpin = 4)
    spinorChiralEmbed(in_full, in[0], chirality);

    // full DiracOverlap::M
    overlap->M(out_full, in_full);

    // out_full(nSpin = 4) -> out(nSpin = 2)
    spinorChiralProject(out[0], out_full, chirality);
    
    const double rho = 4.0 - 1.0 / (2.0 * (kappa));
    blas::ax((2 * rho), out);

    if(zero_shift != 0.0){
      printfQuda("=====zero_shift chiral_overlap=====\n");
      blas::axpby(zero_shift, in, 1.0, out);
    }
  }

  void DiracChiralOverlap::Mdag(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    printfQuda("Entering DiracChiralOverlap::Mdag\n");
    if(overlap){
      overlap->Mdag(out, in);
    }else{
      errorQuda("DiracChiralOverlap::Mdag: overlap pointer is null");
    }
  }

  void DiracChiralOverlap::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src, cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                             const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    for (auto i = 0u; i < b.size(); i++) {
      src[i] = const_cast<ColorSpinorField &>(b[i]).create_alias();
      sol[i] = x[i].create_alias();
    }
  }

  void DiracChiralOverlap::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b, const QudaSolutionType) const
  {
    // do nothing
    printfQuda("DiracChiralOverlap::reconstruct not implemented!\n");
  }

  void DiracChiralOverlap::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
    
    if (overlap) overlap->prefetch(mem_space, stream);
  }

} // namespace quda