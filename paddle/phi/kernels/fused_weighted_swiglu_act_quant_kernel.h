#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void FusedSpaqKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const paddle::optional<DenseTensor>& prob,
                     const bool using_pow2_scaling,
                     DenseTensor* out,
                     DenseTensor* scale);

}  // namespace phi