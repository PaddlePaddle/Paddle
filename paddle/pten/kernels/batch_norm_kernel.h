
#pragma once

#include <string>
#include "paddle/pten/core/dense_tensor.h"

namespace pten {

template <typename T, typename Context>
void BatchNormKernel(const Context& dev_ctx, const DenseTensor& x, \
                    const DenseTensor& scale, const DenseTensor& bias,
                    const DenseTensor& mean, const DenseTensor& variance,
                    float momentum, float epsilon, const std::string& data_layout,
                    bool is_test, bool use_global_stats, bool trainable_statistics,
                    bool fuse_with_relu,
                    DenseTensor* y, DenseTensor* mean_out, DenseTensor* variance_out,
                    DenseTensor* saved_mean, DenseTensor* saved_variance,
                    DenseTensor* reserve_space);

}