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

#pragma once

#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/kernels/funcs/index_impl.cu.h"
#include "paddle/phi/kernels/sparse/convolution_kernel.h"

namespace phi {
namespace sparse {

// TODO(zhangkaihuo): After the GatherCUDAKernel is migrated to phi, replace
// this kernel with phi::GatherCUDAKernel;
// Vectorization can be used to improve read and write bandwidth
/**
 * brief: gather data from params according to indices
 * params: the inputs
 * indices: the indices you want to gather
 * output: the outputs
 * index_size: the size of indices
 * slice_size: slice size corresponding to each index, here is the channel size
**/
template <typename T, typename IndexT = int>
__global__ void GatherKernel(const T* params,
                             const IndexT* indices,
                             T* output,
                             size_t index_size,
                             size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT gather_i = indices[indices_i];
    int64_t params_i = gather_i * slice_size + slice_i;
    *(output + i) = *(params + params_i);
  }
}

/**
 * brief: scatter add
 * input: the inputs
 * unique_value: refer to UpdateIndexKernel notes
 * out_index: the output feature index
 * non_zero_num: the number of output features
 * rulebook_len: the length of rulebook
 * channels: the output channel size
 * out: the outputs
**/
template <typename T>
__global__ void ScatterKernel(const T* input,
                              const int* unique_value,
                              const int* out_index,
                              const int non_zero_num,
                              const int rulebook_len,
                              const int channels,
                              T* out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num * channels; i += gridDim.x * blockDim.x) {
    int indices_i = i / channels;
    int channels_i = i - indices_i * channels;

    int start = unique_value[indices_i];
    int end = indices_i == non_zero_num - 1 ? rulebook_len
                                            : unique_value[indices_i + 1];
    // max(end-start) = kernel_size
    T sum = static_cast<T>(0);
    for (int j = start; j < end; j++) {
      const int out_feature_i = out_index[j];
      sum += input[out_feature_i * channels + channels_i];
    }
    out[indices_i * channels + channels_i] = sum;
  }
}

template <typename Context>
inline int* SortedAndUniqueIndex(const Context& dev_ctx,
                                 const int* rulebook_ptr,
                                 const int len,
                                 DenseTensor* out_index,
                                 DenseTensor* unique_key,
                                 DenseTensor* unique_value) {
  phi::IndexKernel<int, kps::IdentityFunctor<int>>(
      dev_ctx, out_index, kps::IdentityFunctor<int>());
  phi::IndexKernel<int, kps::IdentityFunctor<int>>(
      dev_ctx, unique_value, kps::IdentityFunctor<int>());

  phi::backends::gpu::GpuMemcpyAsync(unique_key->data<int>(),
                                     rulebook_ptr,
                                     sizeof(int) * len,
#ifdef PADDLE_WITH_HIP
                                     hipMemcpyDeviceToDevice,
#else
                                     cudaMemcpyDeviceToDevice,
#endif
                                     dev_ctx.stream());
// compared with thrust::sort_by_key, thrust::merge_by_key may achieved higher
// performance, but thrust::merge_by_key limited by data size
#ifdef PADDLE_WITH_HIP
  thrust::sort_by_key(thrust::hip::par.on(dev_ctx.stream()),
#else
  thrust::sort_by_key(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                      unique_key->data<int>(),
                      unique_key->data<int>() + len,
                      out_index->data<int>());

  // 4. unique
  thrust::pair<int*, int*> new_end =
#ifdef PADDLE_WITH_HIP
      thrust::unique_by_key(thrust::hip::par.on(dev_ctx.stream()),
#else
      thrust::unique_by_key(thrust::cuda::par.on(dev_ctx.stream()),
#endif
                            unique_key->data<int>(),
                            unique_key->data<int>() + len,
                            unique_value->data<int>());
  return new_end.first;
}

}  // namespace sparse
}  // namespace phi
