// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/sync_batch_norm_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/sync_batch_norm_utils.h"

// sparse header
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SyncBatchNormCooGradKernel(
    const Context& dev_ctx,
    const SparseCooTensor& x,
    const DenseTensor& scale,
    const DenseTensor& bias,
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
  *scale_grad = phi::EmptyLike<T, Context>(dev_ctx, scale);
  *bias_grad = phi::EmptyLike<T, Context>(dev_ctx, bias);
  phi::SyncBatchNormGradKernel<T, Context>(dev_ctx,
                                           x.values(),
                                           scale,
                                           bias,
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

}  // namespace sparse
}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(sync_batch_norm_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SyncBatchNormCooGradKernel,
                   float,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(sync_batch_norm_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SyncBatchNormCooGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
