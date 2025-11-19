#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

// MpAllReduceSumKernel 声明
template <typename T, typename Context>
void MpAllReduceSumKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          DenseTensor* out);

}  // namespace phi
