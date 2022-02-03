

#pragma once

#include <string>
#include "paddle/pten/core/dense_tensor.h"


namespace pten {

template <typename T, typename Context>
void BatchNormGradGradKernel(const Context& dev_ctx,  
                    const DenseTensor& x_grad_grad, const DenseTensor& scale_grad_grad,
                    const DenseTensor& bias_grad_grad, const DenseTensor& y_grad,
                    const DenseTensor& x, \
                    const DenseTensor& scale, const DenseTensor& bias,
                    const DenseTensor& saved_mean, const DenseTensor& saved_variance,
                    paddle::optional<const DenseTensor&> mean,
                    paddle::optional<const DenseTensor&> variance,                    
                    float momentum, float epsilon, const std::string& data_layout,
                    bool is_test, bool use_global_stats, bool trainable_statistics,
                    bool fuse_with_relu,
                    DenseTensor* x_grad, DenseTensor* scale_grad, DenseTensor* y_grad_grad );


} // namespace pten