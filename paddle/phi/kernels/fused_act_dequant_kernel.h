#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void FusedActDequantKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& x_scale,
                           DenseTensor* out);

}