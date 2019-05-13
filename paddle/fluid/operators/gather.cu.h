/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using platform::DeviceContext;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void GatherCUDAKernel(const T* params, const int* indices, T* output,
                                 size_t index_size, size_t slice_size) {
  CUDA_1D_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    int gather_i = indices[indices_i];
    int params_i = gather_i * slice_size + slice_i;
    *(output + i) = *(params + params_i);
  }
}

/**
 * A thin wrapper on gpu tensor
 * Return a new tensor from source tensor, gathered according to index
 * input[src]: type-T source Tensor
 * input[index]: type-int index Tensor (1-D)
 * return: output tensor
 */
template <typename T>
void GPUGather(const platform::DeviceContext& ctx, const Tensor& src,
               const Tensor& index, Tensor* output) {
  // PADDLE_ENFORCE(platform::is_gpu_place(place));
  // check index of shape 1-D
  PADDLE_ENFORCE(index.dims().size() == 1 ||
                 (index.dims().size() == 2 && index.dims()[1] == 1));

  int index_size = index.dims()[0];

  auto src_dims = src.dims();
  framework::DDim output_dims(src_dims);
  output_dims[0] = index_size;

  // slice size
  int slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const T* p_src = src.data<T>();
  // why must be int?
  const int* p_index = index.data<int>();
  T* p_output = output->data<T>();

  int block = 512;
  int n = slice_size * index_size;
  int grid = (n + block - 1) / block;

  GatherCUDAKernel<T><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      p_src, p_index, p_output, index_size, slice_size);
}

}  // namespace operators
}  // namespace paddle
