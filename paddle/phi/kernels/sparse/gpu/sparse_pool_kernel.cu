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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/pooling.h"
#include "paddle/phi/kernels/funcs/sparse/convolution.h"
#include "paddle/phi/kernels/sparse/gpu/convolution.cu.h"
#include "paddle/phi/kernels/sparse/sparse_pool_kernel.h"

namespace phi {
namespace sparse {

template <typename T>
__global__ void MaxPoolCudaKernel(const T* in_features_ptr,
                                  const int* rulebook_ptr,
                                  const int n,
                                  const int rulebook_len,
                                  const int channels,
                                  T* out_features_ptr) {
  phi::funcs::MaxPool<T> max_pool_functor;
  CUDA_KERNEL_LOOP_TYPE(i, n * channels, int64_t) {
    int real_i = i / channels;
    int channel_i = i - real_i * channels;
    int in_i = rulebook_ptr[real_i];
    int out_i = rulebook_ptr[real_i + rulebook_len];
    max_pool_functor.compute(in_features_ptr[in_i * channels + channel_i],
                             &out_features_ptr[out_i * channels + channel_i]);
  }
}

/**
 * x: (N, D, H, W, C)
 * kernel: (D, H, W, C, OC)
 * out: (N, D, H, W, OC)
**/
template <typename T, typename Context>
void MaxPoolKernel(const Context& dev_ctx,
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

  std::vector<int> offsets(kernel_size + 1), counter(kernel_size);
  DenseTensorMeta counter_meta(
      DataType::INT32, {kernel_size}, DataLayout::NCHW);
  DenseTensor counter_per_kernel = phi::Empty(dev_ctx, std::move(counter_meta));
  DenseTensor offsets_per_kernel = phi::Empty(dev_ctx, std::move(counter_meta));
  DenseTensorMeta index_meta(DataType::INT32, {1}, DataLayout::NCHW);
  DenseTensor out_index = phi::Empty(dev_ctx, std::move(index_meta));
  DenseTensor unique_key = phi::Empty(dev_ctx, std::move(index_meta));
  DenseTensor unique_value = phi::Empty(dev_ctx, std::move(index_meta));

  // 1. product rulebook
  int rulebook_len = ProductRuleBook<T, Context>(dev_ctx,
                                                 x,
                                                 real_kernel_sizes,
                                                 paddings,
                                                 dilations,
                                                 strides,
                                                 out_dims,
                                                 false,
                                                 rulebook,
                                                 &counter_per_kernel,
                                                 &offsets_per_kernel,
                                                 &out_index,
                                                 &unique_key,
                                                 &unique_value,
                                                 out,
                                                 &counter,
                                                 &offsets);

  const int* rulebook_ptr = rulebook->data<int>();

  T* out_features_ptr = out->mutable_non_zero_elements()->data<T>();
  const T* in_features_ptr = x.non_zero_elements().data<T>();
// 2. max pool
#ifdef PADDLE_WITH_HIP
  thrust::fill(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::fill(thrust::cuda::par.on(dev_ctx.stream()),
#endif
               out_features_ptr,
               out_features_ptr + out->non_zero_elements().numel(),
               static_cast<T>(-FLT_MAX));
  // TODO(zhangkaihuo) Replacing multiple calls with one kernel may be faster
  for (int i = 0; i < kernel_size; i++) {
    if (counter[i] <= 0) {
      continue;
    }

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
        dev_ctx, counter[i] * in_channels, 1);
    MaxPoolCudaKernel<T><<<config.block_per_grid.x,
                           config.thread_per_block.x,
                           0,
                           dev_ctx.stream()>>>(
        in_features_ptr,
        rulebook_ptr + offsets[i] + rulebook_len,
        counter[i],
        rulebook_len,
        in_channels,
        out_features_ptr);
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(sparse_maxpool,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::MaxPoolKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
