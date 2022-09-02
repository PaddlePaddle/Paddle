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

#include "paddle/phi/kernels/sparse/pool_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/funcs/sparse/convolution.h"
#include "paddle/phi/kernels/sparse/cpu/conv.h"

namespace phi {
namespace sparse {

/**
 * x: (N, D, H, W, C)
 * kernel: (D, H, W, C, OC)
 * out: (N, D, H, W, OC)
 **/
template <typename T, typename IntT = int>
void MaxPoolCooCPUKernel(const CPUContext& dev_ctx,
                         const SparseCooTensor& x,
                         const std::vector<int>& kernel_sizes,
                         const std::vector<int>& paddings,
                         const std::vector<int>& dilations,
                         const std::vector<int>& strides,
                         SparseCooTensor* out,
                         DenseTensor* rulebook,
                         DenseTensor* counter) {
  const auto& x_dims = x.dims();
  int kernel_size = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
  const std::vector<int>& real_kernel_sizes =
      phi::funcs::sparse::PoolResetKernel(kernel_sizes, x_dims[4], x_dims[4]);
  DDim out_dims = {1, 1, 1, 1, 1};
  phi::funcs::sparse::GetOutShape(
      x_dims, real_kernel_sizes, paddings, dilations, strides, &out_dims);
  const int in_channels = real_kernel_sizes[3];

  std::vector<int> counter_per_kernel(kernel_size, 0);

  const T* in_features_ptr = x.values().data<T>();
  // 1. product rule book
  ProductRuleBook<T, CPUContext, IntT>(dev_ctx,
                                       x,
                                       real_kernel_sizes,
                                       paddings,
                                       dilations,
                                       strides,
                                       out_dims,
                                       false,
                                       rulebook,
                                       counter_per_kernel.data());

  UpdateRulebookAndOutIndex<T, CPUContext, IntT>(
      dev_ctx, x, kernel_size, in_channels, out_dims, rulebook, out);

  int rulebook_len = rulebook->dims()[1];
  const IntT* rulebook_ptr = rulebook->data<IntT>();

  counter->Resize({kernel_size});
  int* counter_ptr = dev_ctx.template HostAlloc<int>(counter);
  memcpy(counter_ptr, counter_per_kernel.data(), kernel_size * sizeof(int));

  std::vector<int> offsets(kernel_size + 1);
  phi::funcs::sparse::PrefixSum(counter_ptr, &offsets[0], kernel_size);
  std::vector<bool> out_flags(out->nnz(), false);

  // 2. max pool
  T* out_features_ptr = out->mutable_values()->data<T>();
  phi::funcs::MaxPool<T> max_pool_functor;
  for (int i = 0; i < kernel_size; i++) {
    for (int j = 0; j < counter_ptr[i]; j++) {
      IntT in_i = rulebook_ptr[rulebook_len + offsets[i] + j];
      IntT out_i = rulebook_ptr[rulebook_len * 2 + offsets[i] + j];
      if (!out_flags[out_i]) {
        out_flags[out_i] = true;
        memcpy(&out_features_ptr[out_i * in_channels],
               &in_features_ptr[in_i * in_channels],
               in_channels * sizeof(T));
      } else {
        for (int c = 0; c < in_channels; c++) {
          max_pool_functor.compute(in_features_ptr[in_i * in_channels + c],
                                   &out_features_ptr[out_i * in_channels + c]);
        }
      }
    }
  }
}

template <typename T, typename Context>
void MaxPoolCooKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const std::vector<int>& kernel_sizes,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      const std::vector<int>& strides,
                      SparseCooTensor* out,
                      DenseTensor* rulebook,
                      DenseTensor* counter) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "MaxPoolCooCPUKernel", ([&] {
        MaxPoolCooCPUKernel<T, data_t>(dev_ctx,
                                       x,
                                       kernel_sizes,
                                       paddings,
                                       dilations,
                                       strides,
                                       out,
                                       rulebook,
                                       counter);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(maxpool_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MaxPoolCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
