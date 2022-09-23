// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/batch_norm_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void BatchNormInferKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& scale,
                          const DenseTensor& bias,
                          const DenseTensor& mean,
                          const DenseTensor& variance,
                          float momentum,
                          float epsilon,
                          const std::string& data_layout,
                          DenseTensor* y,
                          DenseTensor* mean_out,
                          DenseTensor* variance_out) {
  // Since saved_mean and saved_variance are used regardless of whether
  // they are in test mode, temporary variables need to be created here
  // to be compatible
  auto saved_mean = phi::EmptyLike<T, Context>(dev_ctx, *mean_out);
  auto saved_variance = phi::EmptyLike<T, Context>(dev_ctx, *variance_out);
  BatchNormKernel<T, Context>(dev_ctx,
                              x,
                              scale,
                              bias,
                              mean,
                              variance,
                              momentum,
                              epsilon,
                              data_layout,
                              /*is_test=*/true,
                              /*use_global_stats=*/false,
                              /*trainable_statistics=*/false,
                              /*fuse_with_relu=*/false,
                              y,
                              mean_out,
                              variance_out,
                              &saved_mean,
                              &saved_variance,
                              /*reserve_space=*/nullptr);
}

}  // namespace phi

PD_REGISTER_KERNEL(batch_norm_infer,
                   CPU,
                   ALL_LAYOUT,
                   phi::BatchNormInferKernel,
                   float,
                   double) {}
#ifdef PADDLE_WITH_CUDA
PD_REGISTER_KERNEL(batch_norm_infer,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormInferKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif
#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(batch_norm_infer,
                   GPU,
                   ALL_LAYOUT,
                   phi::BatchNormInferKernel,
                   float,
                   phi::dtype::float16) {}
#endif
