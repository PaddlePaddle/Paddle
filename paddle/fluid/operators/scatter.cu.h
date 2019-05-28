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
#include <unordered_set>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

const int32_t BAD_INDEX = -1;
using Tensor = framework::Tensor;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T, typename IndexT = int>
__global__ void ScatterCUDAKernel(const T* params, const IndexT* indices,
                                  T* output, size_t index_size,
                                  size_t slice_size, bool overwrite) {
  CUDA_1D_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT scatter_i = indices[indices_i];
    IndexT out_i = scatter_i * slice_size + slice_i;
    if (overwrite) {
      if (scatter_i != BAD_INDEX) {
        *(output + out_i) = *(params + i);
      }
    } else {
      paddle::platform::CudaAtomicAdd(output + out_i, *(params + i));
    }
  }
}

/**
 * A thin wrapper on gpu tensor
 * Return a new updated tensor from source tensor, scatter-assigned according to
 * index
 * input[src]: type-T source Tensor
 * input[index]: type-IndexT index Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT = int>
void GPUScatterAssign(const platform::DeviceContext& ctx, const Tensor& src,
                      const Tensor& index, Tensor* output,
                      bool overwrite = true) {
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
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();
  IndexT* uniq_index = new IndexT(index_size);

  // If in overwirte mode, when has same indices, need to
  // unique the indices, because the GPU can not control the random write
  if (overwrite) {
    int unique_index_size = 0;
    std::unordered_set<IndexT> count_index;
    for (int i = 0; i < index_size; ++i) {
      const IndexT& index = p_index[i];
      if (count_index.count(index) > 0) {
        // if find duplidate index, set bad_index
        *(uniq_index + i) = BAD_INDEX;
      } else {
        *(uniq_index + i) = index;
        count_index.emplace(index);
      }
    }
  } else {
    // if in accumulate mode, need init the output
    // init p_output
    memset(p_output, 0, output->numel() * sizeof(T));
  }

  int block = 512;
  int n = slice_size * index_size;
  int grid = (n + block - 1) / block;

  ScatterCUDAKernel<T, IndexT><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      p_src, uniq_index, p_output, index_size, slice_size, overwrite);

  // free memory
  delete[] uniq_index;
}

}  // namespace operators
}  // namespace paddle
