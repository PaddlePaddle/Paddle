/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/batch_norm_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/batch_norm_grad_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi::sparse {

template <typename T, typename Context>
void BatchNormCooGradKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& scale,
                            const DenseTensor& bias,
                            const paddle::optional<DenseTensor>& mean,
                            const paddle::optional<DenseTensor>& variance,
                            const DenseTensor& saved_mean,
                            const DenseTensor& saved_variance,
                            const paddle::optional<DenseTensor>& reserve_space,
                            const SparseCooTensor& y_grad,
                            float momentum,
                            float epsilon,
                            const std::string& data_layout,
                            bool is_test,
                            bool use_global_stats,
                            bool trainable_statistics,
                            SparseCooTensor* x_grad,
                            DenseTensor* scale_grad,
                            DenseTensor* bias_grad) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, x_grad);

  // TODO(umiswing): add check for parameter freezing automatically
  PADDLE_ENFORCE_EQ((scale_grad == nullptr && bias_grad == nullptr) ||
                        (scale_grad != nullptr && bias_grad != nullptr),
                    true,
                    common::errors::InvalidArgument(
                        "Weight and bias's stop_gradient of BatchNorm must be "
                        "True or False at the same time."));

  if (scale_grad && bias_grad) {
    *scale_grad = phi::EmptyLike<T, Context>(dev_ctx, scale);
    *bias_grad = phi::EmptyLike<T, Context>(dev_ctx, bias);
  }
  phi::BatchNormGradKernel<T, Context>(dev_ctx,
                                       x.values(),
                                       scale,
                                       bias,
                                       mean,
                                       variance,
                                       saved_mean,
                                       saved_variance,
                                       reserve_space,
                                       y_grad.values(),
                                       momentum,
                                       epsilon,
                                       data_layout,
                                       is_test,
                                       use_global_stats,
                                       trainable_statistics,
                                       x_grad->mutable_values(),
                                       scale_grad,
                                       bias_grad);
}

}  // namespace phi::sparse

PD_REGISTER_KERNEL(batch_norm_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::BatchNormCooGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

#if defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(batch_norm_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::BatchNormCooGradKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
#endif

#if defined(PADDLE_WITH_CUDA)
PD_REGISTER_KERNEL(batch_norm_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::BatchNormCooGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);  // x_grad
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}
#endif
