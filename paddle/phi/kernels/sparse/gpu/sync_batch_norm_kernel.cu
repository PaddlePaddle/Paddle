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

#include "paddle/phi/kernels/sparse/sync_batch_norm_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sync_batch_norm_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SyncBatchNormCooKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& scale,
                            const DenseTensor& bias,
                            const DenseTensor& mean,
                            const DenseTensor& variance,
                            float momentum,
                            float epsilon,
                            const std::string& data_layout,
                            bool is_test,
                            bool use_global_stats,
                            bool trainable_statistics,
                            bool fuse_with_relu,
                            SparseCooTensor* y,
                            DenseTensor* mean_out,
                            DenseTensor* variance_out,
                            DenseTensor* saved_mean,
                            DenseTensor* saved_variance,
                            DenseTensor* reserve_space) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, y);
  phi::SyncBatchNormKernel<T, Context>(dev_ctx,
                                       x.values(),
                                       scale,
                                       bias,
                                       mean,
                                       variance,
                                       momentum,
                                       epsilon,
                                       data_layout,
                                       is_test,
                                       use_global_stats,
                                       trainable_statistics,
                                       fuse_with_relu,
                                       y->mutable_values(),
                                       mean_out,
                                       variance_out,
                                       saved_mean,
                                       saved_variance,
                                       reserve_space);
  y->SetIndicesDict(x.GetIndicesDict());
}

}  // namespace sparse
}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(sync_batch_norm_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SyncBatchNormCooKernel,
                   float,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(sync_batch_norm_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SyncBatchNormCooKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
