#pragma once

#include <quda_internal.h>
#include <quda.h>

namespace quda
{
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);
  void sinkProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result);
} // namespace quda
