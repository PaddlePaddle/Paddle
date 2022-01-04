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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/place.h"
namespace paddle {
namespace operators {

using framework::Tensor;
using platform::DeviceContext;

template <typename T, typename IndexT = int>
__global__ void GatherCUDAKernel(const T* params, const IndexT* indices,
                                 T* output, size_t index_size,
                                 size_t slice_size) {
  CUDA_KERNEL_LOOP_TYPE(i, index_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT gather_i = indices[indices_i];
    int64_t params_i = gather_i * slice_size + slice_i;
    *(output + i) = *(params + params_i);
  }
}

template <typename T, typename IndexT = int>
__global__ void GatherNdCUDAKernel(const T* input, const int64_t* input_dims,
                                   const IndexT* indices, T* output,
                                   size_t remain_size, size_t slice_size,
                                   size_t end_size) {
  CUDA_KERNEL_LOOP_TYPE(i, remain_size * slice_size, int64_t) {
    int64_t indices_i = i / slice_size;
    int64_t slice_i = i - indices_i * slice_size;  // offset inside the slice
    int64_t gather_i = 0;
    int64_t temp = slice_size;
    for (int64_t j = end_size - 1; j >= 0; --j) {
      auto index_value = indices[indices_i * end_size + j];
      PADDLE_ENFORCE(
          index_value >= 0 && index_value < input_dims[j],
          "The index is out of bounds, "
          "please check whether the dimensions of index and "
          "input meet the requirements. It should "
          "be less than [%d] and greater than or equal to 0, but received [%d]",
          input_dims[j], index_value);
      gather_i += (index_value * temp);
      temp *= input_dims[j];
    }
    int64_t input_i = gather_i + slice_i;
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
  if (index.dims().size() == 2) {
    PADDLE_ENFORCE_EQ(index.dims()[1], 1,
                      platform::errors::InvalidArgument(
                          "If the index's rank of gather_op is 2,"
                          " the second dimension should be 1."));
  }

  // index size
  int64_t index_size = index.dims()[0];
  if (index_size == 0) return;

  auto src_dims = src.dims();
  framework::DDim output_dims(src_dims);
  output_dims[0] = index_size;

  // slice size
  int64_t slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  int block = 512;
  int64_t n = slice_size * index_size;
  int64_t grid = (n + block - 1) / block;

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
  std::vector<int64_t> v_input_dims(input_dims_size);
  for (int i = 0; i < input_dims_size; ++i) {
    v_input_dims[i] = input_dims[i];
  }

  auto& dev_ctx = context.cuda_device_context();
  int64_t bytes = input_dims_size * sizeof(int64_t);
  auto p_input_dims = memory::Alloc(dev_ctx, bytes);
  int64_t* g_input_dims = reinterpret_cast<int64_t*>(p_input_dims->ptr());
  memory::Copy(gplace, g_input_dims, cplace, v_input_dims.data(), bytes,
               ctx.stream());

  int block = 512;
  int64_t n = slice_size * remain_numel;
  int64_t grid = (n + block - 1) / block;

  GatherNdCUDAKernel<T, IndexT><<<
      grid, block, 0,
      reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
      p_input, g_input_dims, p_index, p_output, remain_numel, slice_size,
      end_size);
}

template <typename T, typename U>
__global__ void GatherGPUKernel(const T* input, const U* index, T* out,
                                int64_t outer_dim_size, int64_t inner_dim_size,
                                int64_t out_index_dim_size,
                                int64_t input_index_dim_size, int64_t size) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  int64_t outer_size = outer_dim_size * out_index_dim_size;
  for (; idx < size; idx += blockDim.x * gridDim.x) {
    int64_t inner_dim_index = idx / outer_size;
    int64_t next_idx = idx - outer_size * inner_dim_index;
    int64_t index_dim_index = next_idx / outer_dim_size;
    U index_val = index[index_dim_index];

    PADDLE_ENFORCE(
        index_val >= 0 && index_val < input_index_dim_size,
        "The index is out of bounds, "
        "please check whether the dimensions of index and "
        "input meet the requirements. It should "
        "be less than [%d] and greater than or equal to 0, but received [%d]",
        input_index_dim_size, index_val);

    int64_t out_dim_index = next_idx - outer_dim_size * index_dim_index;
    int64_t input_index =
        inner_dim_index * (outer_dim_size * input_index_dim_size) +
        index_val * outer_dim_size + out_dim_index;
    out[idx] = input[input_index];
  }
}

template <typename T, typename U>
__global__ void GatherGradGPUKernel(const T* input, const U* index, T* out,
                                    int64_t outer_dim_size,
                                    int64_t inner_dim_size,
                                    int64_t input_index_dim_size,
                                    int64_t out_index_dim_size, int64_t size) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < size; idx += blockDim.x * gridDim.x) {
    int64_t inner_dim_index = idx / (outer_dim_size * input_index_dim_size);
    int64_t next_idx = idx % (outer_dim_size * input_index_dim_size);
    int64_t index_dim_index = next_idx / (outer_dim_size);
    int64_t out_dim_index = next_idx % outer_dim_size;
    int64_t out_index =
        inner_dim_index * (outer_dim_size * out_index_dim_size) +
        index[index_dim_index] * outer_dim_size + out_dim_index;
    paddle::platform::CudaAtomicAdd(out + out_index, *(input + idx));
  }
}

template <typename T, typename U>
void GatherV2CUDAFunction(const Tensor* input, const Tensor* index,
                          const int axis, Tensor* out,
                          const paddle::platform::Place& place,
                          const framework::ExecutionContext& ctx) {
  int64_t index_size = index->numel();
  int64_t input_size = input->numel();
  auto input_dim = input->dims();
  auto* input_data = input->data<T>();
  auto* index_data = index->data<U>();

  if (input->numel() == 0) return;

  int axis_index = axis;
  int64_t index_dim_size = input_dim[axis_index];

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  std::vector<int64_t> out_dim_vec;

  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
    out_dim_vec.push_back(input_dim[i]);
  }
  out_dim_vec.push_back(index_size);
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
    out_dim_vec.push_back(input_dim[i]);
  }
  auto out_dim = framework::make_ddim(out_dim_vec);

  out->Resize(out_dim);
  auto* out_data = out->mutable_data<T>(place);
  int64_t out_size = out->numel();
  if (out_size == 0) return;

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), out_size);
  auto stream = ctx.cuda_device_context().stream();
  GatherGPUKernel<
      T, U><<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
      input_data, index_data, out_data, outer_dim_size, inner_dim_size,
      index_size, index_dim_size, out_size);
}

template <typename T, typename U>
void GatherV2GradCUDAFunction(const Tensor* input, const Tensor* index,
                              const int axis, Tensor* out,
                              const paddle::platform::Place& place,
                              const framework::ExecutionContext& ctx) {
  auto* index_data = index->data<U>();
  int64_t index_size = index->numel();
  int64_t input_size = input->numel();
  auto input_dim = input->dims();
  auto* input_data = input->data<T>();

  if (input->numel() == 0) return;
  int axis_index = axis;
  int64_t input_index_dim_size = input_dim[axis_index];

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;

  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
  }
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
  }

  auto* out_data = out->mutable_data<T>(place);
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  auto out_dim = out->dims();
  int64_t out_index_dim_size = out_dim[axis_index];
  operators::math::set_constant(*dev_ctx, out, 0.0);

  platform::GpuLaunchConfig config =
      platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), input_size);
  auto stream = ctx.cuda_device_context().stream();
  GatherGradGPUKernel<
      T, U><<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
      input_data, index_data, out_data, outer_dim_size, inner_dim_size,
      input_index_dim_size, out_index_dim_size, input_size);
}
}  // namespace operators
}  // namespace paddle
