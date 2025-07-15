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
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_resources.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/index_select_impl.h"
#include "paddle/phi/kernels/funcs/repeat_tensor2index_tensor.h"
#include "paddle/phi/kernels/gpu/index_select_impl.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"

namespace phi {

using phi::PADDLE_CUDA_NUM_THREADS;
template <typename T, typename IndexT>
__global__ void index_select_cuda_kernel(const T* input,
                                         T* output,
                                         const IndexT* index,
                                         int64_t N,
                                         int64_t stride,
                                         int64_t size,
                                         int64_t delta) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  const int64_t stride_size = stride * size;

  const int64_t pre_idx = idx / stride_size;
  const int64_t remainder = idx % stride_size;
  const int64_t dim_idx = remainder / stride;

  const IndexT src_dim_idx = index[dim_idx];

  const int64_t input_idx =
      idx + ((delta * pre_idx) + (src_dim_idx - dim_idx)) * stride;
  output[idx] = input[input_idx];
}

template <typename T, typename Context>
void RepeatInterleaveWithTensorIndexKernel(const Context& dev_ctx,
                                           const DenseTensor& x,
                                           const DenseTensor& repeats_tensor,
                                           int dim,
                                           DenseTensor* out) {
  auto input_dim = x.dims();
  if (dim < 0) {
    dim += input_dim.size();
  }
  DenseTensor index;
  PADDLE_ENFORCE_EQ(repeats_tensor.dims()[0] == x.dims()[dim],
                    true,
                    common::errors::InvalidArgument(
                        "The length of Input(RepeatsTensor) must be the "
                        "same as length of Input(X) in axis. "
                        "But received: [%s], required: [%d].",
                        repeats_tensor.dims()[0],
                        x.dims()[dim]));
  const auto& index_type = repeats_tensor.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      common::errors::InvalidArgument(
          "Input(RepeatsTensor) holds the wrong type, it holds %s, but "
          "desires to be %s or %s",
          DataTypeToString(index_type),
          DataTypeToString(phi::DataType::INT32),
          DataTypeToString(phi::DataType::INT64)));

  if (x.numel() == 0) {
    // infer out shape
    if (index_type == phi::DataType::INT32) {
      phi::funcs::RepeatsTensor2IndexTensorFunctor<Context, int>()(
          dev_ctx, repeats_tensor, &index);

    } else if (index_type == phi::DataType::INT64) {
      phi::funcs::RepeatsTensor2IndexTensorFunctor<Context, int64_t>()(
          dev_ctx, repeats_tensor, &index);
    }
    auto output_dim = common::vectorize(x.dims());
    output_dim[dim] = index.dims()[0];
    out->Resize(common::make_ddim(output_dim));
    dev_ctx.template Alloc<T>(out);
    return;
  }

  auto stride_dim = common::stride(input_dim);
  int64_t stride = stride_dim[dim];
  auto stream = dev_ctx.stream();
  auto* in_data = x.data<T>();
  if (index_type == phi::DataType::INT64) {
    phi::funcs::RepeatsTensor2IndexTensorFunctor<Context, int64_t>()(
        dev_ctx, repeats_tensor, &index);

    const int64_t* index_data = index.data<int64_t>();
    auto output_dim = common::vectorize(x.dims());
    output_dim[dim] = index.dims()[0];
    out->Resize(common::make_ddim(output_dim));
    T* out_data = dev_ctx.template Alloc<T>(out);
    int64_t numel = out->numel();
    int64_t size = output_dim[dim];
    int64_t delta = input_dim[dim] - size;

    index_select_cuda_kernel<T, int64_t>
        <<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
           PADDLE_CUDA_NUM_THREADS,
           0,
           stream>>>(in_data, out_data, index_data, numel, stride, size, delta);
  } else {
    phi::funcs::RepeatsTensor2IndexTensorFunctor<Context, int>()(
        dev_ctx, repeats_tensor, &index);

    const int* index_data = index.data<int>();
    auto output_dim = common::vectorize(x.dims());
    output_dim[dim] = index.dims()[0];
    out->Resize(common::make_ddim(output_dim));
    T* out_data = dev_ctx.template Alloc<T>(out);
    int64_t numel = out->numel();
    int64_t size = output_dim[dim];
    int64_t delta = input_dim[dim] - size;
    index_select_cuda_kernel<T, int>
        <<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
           PADDLE_CUDA_NUM_THREADS,
           0,
           stream>>>(in_data, out_data, index_data, numel, stride, size, delta);
  }
}

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
void RepeatInterleaveKernel(const Context& dev_ctx,
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
                   phi::RepeatInterleaveKernel,
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
