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

#include "paddle/phi/kernels/sparse/batch_norm_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi::sparse {

template <typename T, typename Context>
void BatchNormCooKernel(const Context& dev_ctx,
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
  phi::BatchNormKernel<T, Context>(dev_ctx,
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

}  // namespace phi::sparse

PD_REGISTER_KERNEL(batch_norm_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::BatchNormCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

#if defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(batch_norm_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::BatchNormCooKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
}
#endif

#if defined(PADDLE_WITH_CUDA)
PD_REGISTER_KERNEL(batch_norm_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::BatchNormCooKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif
