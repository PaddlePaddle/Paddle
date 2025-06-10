#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void FusedSwigluWeightedBwdKernel(const Context& dev_ctx,
                                const DenseTensor& o1,
                                const DenseTensor& do2_s,
                                const DenseTensor& unzipped_probs,
                                DenseTensor* do1,
                                DenseTensor* probs_grad,
                                DenseTensor* o2_s);

}  // namespace phi