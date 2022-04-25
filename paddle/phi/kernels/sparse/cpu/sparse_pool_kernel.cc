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

#include "paddle/phi/kernels/sparse/sparse_pool_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/funcs/sparse/convolution.h"
#include "paddle/phi/kernels/sparse/cpu/convolution.h"

namespace phi {
namespace sparse {

/**
 * x: (N, D, H, W, C)
 * kernel: (D, H, W, C, OC)
 * out: (N, D, H, W, OC)
**/
template <typename T, typename IntT = int>
void MaxPoolCPUKernel(const CPUContext& dev_ctx,
                      const SparseCooTensor& x,
                      const std::vector<int>& kernel_sizes,
                      const std::vector<int>& paddings,
                      const std::vector<int>& dilations,
                      const std::vector<int>& strides,
                      SparseCooTensor* out,
                      DenseTensor* rulebook) {
  const auto& x_dims = x.dims();
  int kernel_size = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
  const std::vector<int>& real_kernel_sizes =
      phi::funcs::sparse::PoolResetKernel(kernel_sizes, x_dims[4], x_dims[4]);
  DDim out_dims = {1, 1, 1, 1, 1};
  phi::funcs::sparse::GetOutShape(
      x_dims, real_kernel_sizes, paddings, dilations, strides, &out_dims);
  const int in_channels = real_kernel_sizes[3];

  DenseTensorMeta counter_meta(
      DataType::INT32, {kernel_size}, DataLayout::NCHW);
  DenseTensor counter_per_kernel = phi::Empty(dev_ctx, std::move(counter_meta));

  const T* in_features_ptr = x.non_zero_elements().data<T>();
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
                                       &counter_per_kernel);

  UpdateRulebookAndOutIndex<T, CPUContext, IntT>(
      dev_ctx, x, kernel_size, in_channels, out_dims, rulebook, out);

  int rulebook_len = rulebook->dims()[1];
  const IntT* rulebook_ptr = rulebook->data<IntT>();
  const int* counter_ptr = counter_per_kernel.data<int>();

  std::vector<int> offsets(kernel_size + 1);
  phi::funcs::sparse::PrefixSum(counter_ptr, &offsets[0], kernel_size);
  std::vector<bool> out_flags(out->nnz(), false);

  // 2. max pool
  T* out_features_ptr = out->mutable_non_zero_elements()->data<T>();
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
void MaxPoolKernel(const Context& dev_ctx,
                   const SparseCooTensor& x,
                   const std::vector<int>& kernel_sizes,
                   const std::vector<int>& paddings,
                   const std::vector<int>& dilations,
                   const std::vector<int>& strides,
                   SparseCooTensor* out,
                   DenseTensor* rulebook) {
  PD_VISIT_INTEGRAL_TYPES(
      x.non_zero_indices().dtype(), "MaxPoolCPUKernel", ([&] {
        MaxPoolCPUKernel<T, data_t>(dev_ctx,
                                    x,
                                    kernel_sizes,
                                    paddings,
                                    dilations,
                                    strides,
                                    out,
                                    rulebook);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_maxpool,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::MaxPoolKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
