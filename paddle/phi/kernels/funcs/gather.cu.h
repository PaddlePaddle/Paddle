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

#include "paddle/phi/common/memory_utils.h"
// TODO(paddle-dev): move gpu_primitives.h to phi
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {
namespace funcs {

template <typename T, typename IndexT, int VecSize>
__global__ void GatherNdCUDAKernel(const T* input,
                                   const Dim<DDim::kMaxRank> input_dims,
                                   const IndexT* indices,
                                   T* output,
                                   size_t remain_size,
                                   size_t slice_size,
                                   size_t end_size) {
  size_t total_size = remain_size * slice_size;
  size_t idx =
      (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VecSize;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x * VecSize;

#pragma unroll
  for (; idx < total_size; idx += stride) {
    size_t indices_i = idx / slice_size;
    size_t slice_i = idx % slice_size;
    size_t gather_i = 0;
    size_t gather_stride = slice_size;
#pragma unroll
    for (int j = end_size - 1; j >= 0; --j) {
      auto index_value = indices[indices_i * end_size + j];
      PADDLE_ENFORCE(
          index_value >= -input_dims[j] && index_value < input_dims[j],
          "The index is out of bounds, "
          "please check whether the dimensions of index and "
          "input meet the requirements. It should "
          "be less than [%ld] and greater than or equal to [%ld], but "
          "received [%ld]",
          input_dims[j],
          -input_dims[j],
          index_value);
      if (index_value < 0) {
        index_value += input_dims[j];
      }
      gather_i += index_value * gather_stride;
      gather_stride *= input_dims[j];
    }
    size_t input_i = gather_i + slice_i;

    using VecType = kps::details::VectorType<T, VecSize>;
    const VecType* src = reinterpret_cast<const VecType*>(&input[input_i]);
    VecType* dst = reinterpret_cast<VecType*>(&output[idx]);
    *dst = *src;
  }
}

template <typename T, typename IndexT = int>
void GPUGatherNd(const phi::GPUContext& ctx,
                 const DenseTensor& input,
                 const DenseTensor& index,
                 DenseTensor* output) {
  const auto gplace = ctx.GetPlace();
  auto cplace = phi::CPUPlace();

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
  auto remain_ddim = common::slice_ddim(index_dims, 0, index_dims_size - 1);
  int64_t remain_numel = common::product(remain_ddim);
  // slice size
  int64_t slice_size = 1;
  for (int64_t i = end_size; i < input_dims_size; ++i) {
    slice_size *= input_dims[i];
  }
  // source dim
  Dim<DDim::kMaxRank> g_input_dims;
  for (int i = 0; i < input_dims_size; ++i) {
    g_input_dims[i] = input_dims[i];
  }

  int vec_size = 4;
  vec_size = std::min(phi::GetVectorizedSize(p_input), vec_size);
  vec_size = std::min(phi::GetVectorizedSize(p_output), vec_size);
  while (vec_size > 1 && slice_size % vec_size != 0) {
    vec_size /= 2;
  }

  constexpr int loop_count = 4;
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      ctx, remain_numel * slice_size, vec_size * loop_count);

  auto stream = ctx.stream();

  switch (vec_size) {
#define CASE_VEC_SIZE(__Sz)                                              \
  case __Sz:                                                             \
    GatherNdCUDAKernel<T, IndexT, __Sz>                                  \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            p_input,                                                     \
            g_input_dims,                                                \
            p_index,                                                     \
            p_output,                                                    \
            remain_numel,                                                \
            slice_size,                                                  \
            end_size);                                                   \
    break
    CASE_VEC_SIZE(4);
    CASE_VEC_SIZE(2);
    CASE_VEC_SIZE(1);
#undef CASE_VEC_SIZE
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported vectorized size: %d", vec_size));
  }
}

template <typename T, typename U, int VecSize>
__global__ void GatherGPUKernel(const T* input,
                                const U* index,
                                T* out,
                                int64_t outer_dim_size,
                                int64_t out_index_dim_size,
                                int64_t input_index_dim_size,
                                int64_t size) {
  int64_t block_size = blockDim.x;
  int64_t idx = (blockIdx.x * block_size + threadIdx.x) * VecSize;
  int64_t outer_size = outer_dim_size * out_index_dim_size;
  for (; idx < size; idx += gridDim.x * block_size * VecSize) {
    int64_t inner_dim_index = idx / outer_size;
    int64_t next_idx = idx % outer_size;
    int64_t index_dim_index = next_idx / outer_dim_size;
    U index_val = index[index_dim_index];

    PADDLE_ENFORCE(
        index_val >= -input_index_dim_size && index_val < input_index_dim_size,
        "The index is out of bounds, "
        "please check whether the dimensions of index and "
        "input meet the requirements. It should "
        "be less than [%ld] and greater than or equal to [%ld], but "
        "received [%ld]",
        input_index_dim_size,
        -input_index_dim_size,
        index_val);
    if (index_val < 0) {
      index_val += input_index_dim_size;
    }

    int64_t out_dim_index = next_idx % outer_dim_size;
    int64_t input_index =
        inner_dim_index * (outer_dim_size * input_index_dim_size) +
        index_val * outer_dim_size + out_dim_index;

    using VecType = kps::details::VectorType<T, VecSize>;
    const VecType* src = reinterpret_cast<const VecType*>(&input[input_index]);
    VecType* dst = reinterpret_cast<VecType*>(&out[idx]);
    *dst = *src;
  }
}

template <typename T, typename U>
__global__ void GatherGradGPUKernel(const T* input,
                                    const U* index,
                                    T* out,
                                    int64_t outer_dim_size,
                                    int64_t inner_dim_size,
                                    int64_t input_index_dim_size,
                                    int64_t out_index_dim_size,
                                    int64_t size) {
  int64_t idx = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; idx < size; idx += stride) {
    int64_t inner_dim_index = idx / (outer_dim_size * input_index_dim_size);
    int64_t next_idx = idx % (outer_dim_size * input_index_dim_size);
    int64_t index_dim_index = next_idx / (outer_dim_size);
    int64_t out_dim_index = next_idx % outer_dim_size;
    int64_t out_index =
        inner_dim_index * (outer_dim_size * out_index_dim_size) +
        index[index_dim_index] * outer_dim_size + out_dim_index;
    phi::CudaAtomicAdd(out + out_index, *(input + idx));
  }
}

template <typename T, typename U>
void GatherV2CUDAFunction(const DenseTensor* input,
                          const DenseTensor* index,
                          const int axis,
                          DenseTensor* out,
                          const phi::GPUContext& ctx) {
  int64_t index_size = index->numel();
  int64_t input_size = input->numel();
  auto input_dim = input->dims();
  auto* input_data = input->data<T>();
  auto* index_data = index->data<U>();

  if (input->numel() == 0) return;

  int axis_index = axis;
  int64_t index_dim_size = input_dim[axis_index];

  int64_t outer_dim_size = 1;
  std::vector<int64_t> out_dim_vec;

  for (int i = 0; i < axis_index; i++) {
    out_dim_vec.push_back(input_dim[i]);
  }
  if (index->dims().size() != 0) {
    out_dim_vec.push_back(index_size);
  }
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
    out_dim_vec.push_back(input_dim[i]);
  }
  auto out_dim = common::make_ddim(out_dim_vec);

  out->Resize(out_dim);
  auto* out_data = ctx.Alloc<T>(out);
  int64_t out_size = out->numel();
  if (out_size == 0) return;

  int vec_size = 4;
  vec_size = std::min(phi::GetVectorizedSize(input), vec_size);
  vec_size = std::min(phi::GetVectorizedSize(out), vec_size);
  while (vec_size > 1 && outer_dim_size % vec_size != 0) {
    vec_size /= 2;
  }

  constexpr int loop_count = 4;
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      ctx, out_size, vec_size * loop_count);
  auto stream = ctx.stream();

  switch (vec_size) {
#define CASE_VEC_SIZE(__Sz)                                              \
  case __Sz:                                                             \
    GatherGPUKernel<T, U, __Sz>                                          \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            input_data,                                                  \
            index_data,                                                  \
            out_data,                                                    \
            outer_dim_size,                                              \
            index_size,                                                  \
            index_dim_size,                                              \
            out_size);                                                   \
    break
    CASE_VEC_SIZE(4);
    CASE_VEC_SIZE(2);
    CASE_VEC_SIZE(1);
#undef CASE_VEC_SIZE
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported vectorized size: %d", vec_size));
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
void GPUGather(const phi::GPUContext& ctx,
               const DenseTensor& src,
               const DenseTensor& index,
               DenseTensor* output) {
  GatherV2CUDAFunction<T, IndexT>(&src, &index, /* axis= */ 0, output, ctx);
}

template <typename T, typename U>
void GatherV2GradCUDAFunction(const DenseTensor* input,
                              const DenseTensor* index,
                              const int axis,
                              DenseTensor* out,
                              const phi::GPUContext& ctx) {
  auto* index_data = index->data<U>();
  int64_t index_size = index->numel();
  int64_t input_size = input->numel();
  auto input_dim = input->dims();
  auto* input_data = input->data<T>();

  if (input->numel() == 0) return;
  int axis_index = axis;
  int64_t input_index_dim_size =
      index->dims().size() == 0 ? 1 : input_dim[axis_index];

  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;

  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
  }
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
  }

  auto* out_data = ctx.Alloc<T>(out);
  auto out_dim = out->dims();
  int64_t out_index_dim_size = out_dim[axis_index];
  phi::funcs::set_constant(ctx, out, static_cast<float>(0.0));

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(ctx, input_size);
  auto stream = ctx.stream();
  GatherGradGPUKernel<T, U>
      <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
          input_data,
          index_data,
          out_data,
          outer_dim_size,
          inner_dim_size,
          input_index_dim_size,
          out_index_dim_size,
          input_size);
}

}  // namespace funcs
}  // namespace phi
