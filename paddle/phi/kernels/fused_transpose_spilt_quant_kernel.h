#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void FusedTransposeSplitQuantKernel(const Context& dev_ctx,
                                    const DenseTensor& x,
                                    const std::vector<int64_t>& tokens_per_expert,
                                    bool pow_2_scales,
                                    std::vector<DenseTensor*> outs,
                                    std::vector<DenseTensor*> scales);

}  // namespace phi