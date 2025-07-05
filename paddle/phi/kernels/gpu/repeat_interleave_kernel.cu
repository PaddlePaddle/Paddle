// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/repeat_interleave_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/repeat_interleave_kernel_impl.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {

// Vectorized version for better memory throughput
template <typename T, int VecSize>
__global__ void RepeatInterleaveVecKernel(const T* __restrict__ input,
                                          T* __restrict__ output,
                                          const int64_t numel,
                                          const int64_t outer_size,
                                          const int64_t repeat_size,
                                          const int64_t inner_size,
                                          const int repeats) {
  using VecType = kps::details::VectorType<T, VecSize>;

  const int64_t tid = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  if (tid >= numel) return;

  VecType* vec_output = reinterpret_cast<VecType*>(output);
  const VecType* vec_input = reinterpret_cast<const VecType*>(input);

#pragma unroll
  for (int v = 0; v < VecSize && tid + v < numel; v++) {
    const int64_t idx = tid + v;
    const int64_t inner_idx = idx % inner_size;
    const int64_t temp = idx / inner_size;
    const int64_t repeat_idx = temp % (repeat_size * repeats);
    const int64_t outer_idx = temp / (repeat_size * repeats);
    const int64_t src_repeat_idx = repeat_idx / repeats;
    const int64_t src_idx = outer_idx * repeat_size * inner_size +
                            src_repeat_idx * inner_size + inner_idx;

    if (v == 0 && (idx % VecSize == 0) && ((idx + VecSize) <= numel)) {
      vec_output[idx / VecSize] = vec_input[src_idx / VecSize];
      break;
    } else {
      output[idx] = input[src_idx];
    }
  }
}
template <typename T, typename Context>
void RepeatInterleaveKernelV2(const Context& dev_ctx,
                              const DenseTensor& x,
                              int repeats,
                              int dim,
                              DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (out && out->numel() == 0) {
    return;
  }
  // Get actual dimension
  const int ndim = x.dims().size();
  const int target_dim = (dim < 0) ? ndim + dim : dim;

  // Calculate sizes
  int64_t outer_size = 1;
  for (int i = 0; i < target_dim; i++) {
    outer_size *= x.dims()[i];
  }

  const int64_t repeat_size = x.dims()[target_dim];

  int64_t inner_size = 1;
  for (int i = target_dim + 1; i < ndim; i++) {
    inner_size *= x.dims()[i];
  }

  const int64_t total_elements =
      outer_size * repeat_size * repeats * inner_size;

  int vec_size = 8;
  vec_size = std::min(phi::GetVectorizedSize(x.data<T>()), vec_size);
  vec_size = std::min(phi::GetVectorizedSize(out->data<T>()), vec_size);
  while (vec_size > 1 && inner_size % vec_size != 0) {
    vec_size /= 2;
  }

  constexpr int loop_count = 1;
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(
      dev_ctx, total_elements, vec_size * loop_count);

  switch (vec_size) {
#define CASE_VEC_SIZE(__Sz)                                                  \
  case __Sz:                                                                 \
    RepeatInterleaveVecKernel<T, __Sz><<<config.block_per_grid,              \
                                         config.thread_per_block,            \
                                         0,                                  \
                                         dev_ctx.stream()>>>(x.data<T>(),    \
                                                             out->data<T>(), \
                                                             total_elements, \
                                                             outer_size,     \
                                                             repeat_size,    \
                                                             inner_size,     \
                                                             repeats);       \
    break
    CASE_VEC_SIZE(8);
    CASE_VEC_SIZE(4);
    CASE_VEC_SIZE(2);
    CASE_VEC_SIZE(1);
#undef CASE_VEC_SIZE
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Unsupported vectorized size: %d", vec_size));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(repeat_interleave,
                   GPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveKernelV2,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(repeat_interleave_with_tensor_index,
                   GPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveWithTensorIndexKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
