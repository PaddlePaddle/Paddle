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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/sync_batch_norm_utils.h"
#include "paddle/phi/kernels/sync_batch_norm_kernel.h"

namespace phi {

template <typename T, typename Context>
void SyncBatchNormGradKernel(const Context& ctx,
                             const DenseTensor& x,
                             const DenseTensor& scale,
                             const DenseTensor& bias,
                             const DenseTensor& saved_mean,
                             const DenseTensor& saved_variance,
                             const paddle::optional<DenseTensor>& reserve_space,
                             const DenseTensor& y_grad,
                             float momentum,
                             float epsilon_f,
                             const std::string& data_layout_str,
                             bool is_test,
                             bool use_global_stats,
                             bool trainable_statistics,
                             DenseTensor* x_grad,
                             DenseTensor* scale_grad,
                             DenseTensor* bias_grad) {
  SyncBatchNormGradFunctor<T, Context>(ctx,
                                       &x,
                                       nullptr,
                                       scale,
                                       bias,
                                       saved_mean,
                                       saved_variance,
                                       y_grad,
                                       epsilon_f,
                                       data_layout_str,
                                       x_grad,
                                       scale_grad,
                                       bias_grad);
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(sync_batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SyncBatchNormGradKernel,
                   float,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}
#else
#if CUDNN_VERSION_MIN(8, 1, 0)
PD_REGISTER_KERNEL(sync_batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SyncBatchNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#else
PD_REGISTER_KERNEL(sync_batch_norm_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SyncBatchNormGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
#endif
