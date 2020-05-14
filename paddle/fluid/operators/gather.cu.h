/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "paddle/fluid/framework/dim.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using platform::DeviceContext;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T, typename IndexT = int>
__global__ void GatherCUDAKernel(const T* params, const IndexT* indices,
                                 T* output, size_t index_size,
                                 size_t slice_size) {
  CUDA_1D_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT gather_i = indices[indices_i];
    IndexT params_i = gather_i * slice_size + slice_i;
    *(output + i) = *(params + params_i);
  }
}

template <typename T, typename IndexT = int>
__global__ void GatherNdCUDAKernel(const T* input, const int* input_dims,
                                   const IndexT* indices, T* output,
                                   size_t remain_size, size_t slice_size,
                                   size_t end_size) {
  CUDA_1D_KERNEL_LOOP(i, remain_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT gather_i = 0;
    int64_t temp = slice_size;
    for (int64_t j = end_size - 1; j >= 0; --j) {
      auto index_value = indices[indices_i * end_size + j];
      assert(index_value >= 0 && index_value < input_dims[j]);
      gather_i += (index_value * temp);
      temp *= input_dims[j];
    }
    IndexT input_i = gather_i + slice_i;
    *(output + i) = *(input + input_i);
  }
}

/**
 * A thin wrapper on gpu tensor
 * Return a new tensor from source tensor, gathered according to index
 * input[src]: type-T source Tensor
 * input[index]: type-IndexT index Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT = int>
void GPUGather(const platform::DeviceContext& ctx, const Tensor& src,
               const Tensor& index, Tensor* output) {
  // check index of shape 1-D
  if (index.dims().size() == 1) {
    PADDLE_ENFORCE_GT(index.dims()[0], 0,
                      platform::errors::InvalidArgument(
                          "The index of gather_op should not be empty"
                          "when the index's rank is 1."));
  } else if (index.dims().size() == 2) {
    PADDLE_ENFORCE_EQ(index.dims()[1], 1,
                      platform::errors::InvalidArgument(
                          "If the index's rank of gather_op is 2,"
                          " the second dimension should be 1."));
  }

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

  int block = 512;
  int n = slice_size * index_size;
  int grid = (n + block - 1) / block;

  GatherCUDAKernel<T, IndexT><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      p_src, p_index, p_output, index_size, slice_size);
}

template <typename DeviceContext, typename T, typename IndexT = int>
void GPUGatherNd(const framework::ExecutionContext& context,
                 const Tensor& input, const Tensor& index, Tensor* output) {
  const auto& ctx = context.template device_context<DeviceContext>();
  const auto gplace = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
  auto cplace = platform::CPUPlace();

  auto index_dims = index.dims();
  auto index_dims_size = index_dims.size();
  auto input_dims = input.dims();
  auto input_dims_size = input_dims.size();

  const T* p_input = input.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  // final dim
  int64_t end_size = index_dims[index_dims_size - 1];
  // remain dim
  auto remain_ddim = framework::slice_ddim(index_dims, 0, index_dims_size - 1);
  int64_t remain_numel = framework::product(remain_ddim);
  // slice size
  int64_t slice_size = 1;
  for (int64_t i = end_size; i < input_dims_size; ++i) {
    slice_size *= input_dims[i];
  }
  // source dim
  std::vector<int> v_input_dims(input_dims_size);
  for (int i = 0; i < input_dims_size; ++i) {
    v_input_dims[i] = static_cast<int>(input_dims[i]);
  }

  auto& dev_ctx = context.cuda_device_context();
  int bytes = input_dims_size * sizeof(int);
  auto p_input_dims = memory::Alloc(dev_ctx, bytes);
  int* g_input_dims = reinterpret_cast<int*>(p_input_dims->ptr());
  memory::Copy(gplace, g_input_dims, cplace, v_input_dims.data(), bytes,
               ctx.stream());

  int block = 512;
  int n = slice_size * remain_numel;
  int grid = (n + block - 1) / block;

  GatherNdCUDAKernel<T, IndexT><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      p_input, g_input_dims, p_index, p_output, remain_numel, slice_size,
      end_size);
}

}  // namespace operators
}  // namespace paddle
