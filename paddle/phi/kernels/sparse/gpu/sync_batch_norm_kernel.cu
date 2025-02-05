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

#include "paddle/phi/kernels/sync_batch_norm_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/sync_batch_norm_utils.h"

// sparse header
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SyncBatchNormCooKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& mean,
                            const DenseTensor& variance,
                            const DenseTensor& scale,
                            const DenseTensor& bias,
                            bool is_test,
                            float momentum,
                            float epsilon,
                            const std::string& data_layout,
                            bool use_global_stats,
                            bool trainable_statistics,
                            SparseCooTensor* y,
                            DenseTensor* mean_out,
                            DenseTensor* variance_out,
                            DenseTensor* saved_mean,
                            DenseTensor* saved_variance,
                            DenseTensor* reserve_space) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, y);
  phi::SyncBatchNormKernel<T, Context>(dev_ctx,
                                       x.values(),
                                       mean,
                                       variance,
                                       scale,
                                       bias,
                                       is_test,
                                       momentum,
                                       epsilon,
                                       data_layout,
                                       use_global_stats,
                                       trainable_statistics,
                                       y->mutable_values(),
                                       mean_out,
                                       variance_out,
                                       saved_mean,
                                       saved_variance,
                                       reserve_space);
  y->SetIndicesDict(x.GetIndicesDict());
  y->SetKmaps(x.GetKmaps());
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
