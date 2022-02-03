


#pragma once

#include <string>
#include "paddle/pten/core/dense_tensor.h"


namespace pten {

template <typename T, typename Context>
void BatchNormGradRawKernel(const Context& dev_ctx,  const DenseTensor& y_grad,
                    const DenseTensor& x, \
                    const DenseTensor& scale, const DenseTensor& bias,
                    const DenseTensor& saved_mean, const DenseTensor& saved_variance,
                    paddle::optional<const DenseTensor&> reserve_space,
                    paddle::optional<const DenseTensor&> mean,
                    paddle::optional<const DenseTensor&> variance,
                    float momentum, float epsilon, const std::string& data_layout,
                    bool is_test, bool use_global_stats, bool trainable_statistics,
                    bool fuse_with_relu, bool is_inplace,
                    DenseTensor* x_grad, DenseTensor* scale_grad, DenseTensor* bias_grad );

template <typename T, typename Context>
void BatchNormGradKernel(const Context& dev_ctx,  const DenseTensor& y_grad,
                    const DenseTensor& x, \
                    const DenseTensor& scale, const DenseTensor& bias,
                    const DenseTensor& saved_mean, const DenseTensor& saved_variance,
                    paddle::optional<const DenseTensor&> reserve_space,
                    paddle::optional<const DenseTensor&> mean,
                    paddle::optional<const DenseTensor&> variance,
                    float momentum, float epsilon, const std::string& data_layout,
                    bool is_test, bool use_global_stats, bool trainable_statistics,
                    bool fuse_with_relu,
                    DenseTensor* x_grad, DenseTensor* scale_grad, DenseTensor* bias_grad );


} // namespace pten