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
#include <unordered_set>
#include <vector>
#include "math/math_function.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, typename IndexT = int>
__global__ void ScatterInitCUDAKernel(const IndexT* indices, T* output,
                                      size_t index_size, size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT scatter_i = indices[indices_i];

    PADDLE_ENFORCE(scatter_i >= 0,
                   "The index is out of bounds, "
                   "please check whether the dimensions of index and "
                   "input meet the requirements. It should "
                   "be greater than or equal to 0, but received [%d]",
                   scatter_i);

    int64_t out_i = scatter_i * slice_size + slice_i;
    *(output + out_i) = static_cast<T>(0);
  }
}

template <typename T, typename IndexT = int>
__global__ void ScatterCUDAKernel(const T* params, const IndexT* indices,
                                  T* output, size_t index_size,
                                  size_t slice_size, bool overwrite) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT scatter_i = indices[indices_i];

    PADDLE_ENFORCE(scatter_i >= 0,
                   "The index is out of bounds, "
                   "please check whether the dimensions of index and "
                   "input meet the requirements. It should "
                   "be greater than or equal to 0, but received [%d]",
                   scatter_i);

    int64_t out_i = scatter_i * slice_size + slice_i;
    if (overwrite) {
      *(output + out_i) = *(params + i);
    } else {
      paddle::platform::CudaAtomicAdd(output + out_i, *(params + i));
    }
  }
}

template <typename T, typename IndexT = int>
__global__ void ScatterNdCUDAKernel(const T* update, const IndexT* indices,
                                    T* output, const int64_t* output_dims,
                                    size_t remain_size, size_t slice_size,
                                    size_t end_size) {
  CUDA_KERNEL_LOOP_TYPE(i, remain_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;  // offset inside the slice
    int64_t gather_i = 0;
    int64_t temp = slice_size;
    for (int64_t j = end_size - 1; j >= 0; --j) {
      IndexT index_value = indices[indices_i * end_size + j];

      PADDLE_ENFORCE(
          index_value >= 0 && index_value < output_dims[j],
          "The index is out of bounds, "
          "please check whether the dimensions of index and "
          "input meet the requirements. It should "
          "be less than [%d] and greater or equal to 0, but received [%d]",
          output_dims[j], index_value);

      gather_i += (index_value * temp);
      temp *= output_dims[j];
    }
    int64_t output_i = gather_i + slice_i;
    paddle::platform::CudaAtomicAdd(output + output_i, *(update + i));
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
void GPUScatterAssign(const framework::ExecutionContext& context,
                      const Tensor& src, const Tensor& index, Tensor* output,
                      bool overwrite = true) {
  // check index of shape 1-D
  const auto& ctx = context.device_context();
  if (index.dims().size() == 2) {
    PADDLE_ENFORCE_EQ(index.dims()[1], 1,
                      platform::errors::InvalidArgument(
                          "index.dims()[1] should be 1 when "
                          "index.dims().size() = 2 in scatter_op."
                          "But received value is [%d]",
                          index.dims()[1]));
  } else {
    PADDLE_ENFORCE_EQ(index.dims().size(), 1,
                      platform::errors::InvalidArgument(
                          "index.dims().size() should be 1 or 2 in scatter_op."
                          "But received value is [%d]",
                          index.dims().size()));
  }
  int64_t index_size = index.dims()[0];

  auto src_dims = src.dims();
  framework::DDim output_dims(src_dims);
  output_dims[0] = index_size;

  // slice size
  int64_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();
  const size_t& slice_bytes = slice_size * sizeof(T);

  // set block and grid num
  int block = 512;
  int64_t n = slice_size * index_size;
  int64_t grid = (n + block - 1) / block;

  // if not overwrite mode, init data
  if (!overwrite) {
    ScatterInitCUDAKernel<T, IndexT><<<
        grid, block, 0,
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
        p_index, p_output, index_size, slice_size);
  }

  ScatterCUDAKernel<T, IndexT><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      p_src, p_index, p_output, index_size, slice_size, overwrite);
}

// The function is only for scatter grad x,
// however update grad use gather
template <typename T, typename IndexT = int>
void GPUScatterGradForX(const platform::DeviceContext& ctx, const Tensor& index,
                        Tensor* output) {
  int64_t index_size = index.dims()[0];
  auto dst_dims = output->dims();
  // slice size
  int64_t slice_size = 1;
  for (int i = 1; i < dst_dims.size(); ++i) slice_size *= dst_dims[i];
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();
  const size_t& slice_bytes = slice_size * sizeof(T);

  // set block and grid num
  int64_t block = 512;
  int64_t n = slice_size * index_size;
  int64_t height = (n + block - 1) / block;

  int64_t max_grid_dimx =
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx)
          .GetCUDAMaxGridDimSize()
          .x;
  int64_t grid = height < max_grid_dimx ? height : max_grid_dimx;

  ScatterInitCUDAKernel<T, IndexT><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      p_index, p_output, index_size, slice_size);
}

template <typename DeviceContext, typename T, typename IndexT = int>
void GPUScatterNdAdd(const framework::ExecutionContext& context,
                     const Tensor& update, const Tensor& index,
                     Tensor* output) {
  auto index_dims = index.dims();
  auto index_dims_size = index_dims.size();

  auto output_dims = output->dims();
  auto output_dims_size = output_dims.size();

  const T* p_update = update.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  // final dim
  int64_t end_size = index_dims[index_dims_size - 1];
  // remain dim
  auto remain_ddim = framework::slice_ddim(index_dims, 0, index_dims_size - 1);
  int64_t remain_numel = framework::product(remain_ddim);
  // slice size
  int64_t slice_size = 1;
  for (int64_t i = end_size; i < output_dims_size; ++i) {
    slice_size *= output_dims[i];
  }
  const size_t slice_bytes = slice_size * sizeof(T);
  // put output_dims int CUDA
  // gplace and cplace
  const auto& ctx = context.template device_context<DeviceContext>();
  const auto gplace = ctx.GetPlace();
  auto cplace = platform::CPUPlace();

  std::vector<int64_t> v_output_dims(output_dims_size);
  for (int i = 0; i < output_dims_size; ++i) {
    v_output_dims[i] = output_dims[i];
  }
  auto& dev_ctx = context.cuda_device_context();
  int64_t bytes = output_dims_size * sizeof(int64_t);
  auto output_dims_ptr = memory::Alloc(dev_ctx, bytes);
  int64_t* g_output_dims = reinterpret_cast<int64_t*>(output_dims_ptr->ptr());
  memory::Copy(gplace, g_output_dims, cplace, v_output_dims.data(), bytes,
               ctx.stream());

  int block = 512;
  int64_t n = slice_size * remain_numel;
  int64_t grid = (n + block - 1) / block;

  ScatterNdCUDAKernel<T, IndexT><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      p_update, p_index, p_output, g_output_dims, remain_numel, slice_size,
      end_size);
}

}  // namespace operators
}  // namespace paddle
