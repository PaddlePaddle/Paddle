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

#include "paddle/phi/kernels/repeat_interleave_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/cpu/index_select_impl.h"
#include "paddle/phi/kernels/funcs/repeat_tensor2index_tensor.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#include "paddle/phi/kernels/primitive/kernel_primitives.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#ifdef __NVCC__
#include "cub/cub.cuh"
#else
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
namespace phi {
using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename IndexT>
__global__ void index_select_grad_cuda_kernel(const T* output_grad,
                                              T* input_grad,
                                              const IndexT* index,
                                              int64_t output_grad_numel,
                                              int64_t stride,
                                              int64_t size,
                                              int64_t delta) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= output_grad_numel) {
    return;
  }

  int64_t pre_idx = idx / (stride * size);
  int64_t dim_idx = idx % (stride * size) / stride;
  IndexT src_dim_idx = index[dim_idx];
  int64_t input_idx = idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
  phi::CudaAtomicAdd(&input_grad[input_idx], output_grad[idx]);
}

template <typename T, int VecSize>
__global__ void index_select_grad_init(T* input_grad, int64_t numel) {
  using VecType = kps::details::VectorType<T, VecSize>;

  const int64_t tid = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
  if (tid >= numel) return;

  T set_value[VecSize];
#pragma unroll
  for (int i = 0; i < VecSize; i++) {
    set_value[i] = 0;
  }
  const VecType* vec_value = reinterpret_cast<const VecType*>(&set_value[0]);

#pragma unroll
  for (int64_t i = tid; i < numel; i += blockDim.x * gridDim.x * VecSize) {
    VecType* vec_output = reinterpret_cast<VecType*>(&input_grad[tid]);
    *vec_output = *vec_value;
  }
}
template <typename T, typename Context>
void RepeatInterleaveWithTensorIndexGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& repeats_tensor,
    const DenseTensor& out_grad,
    int dim,
    DenseTensor* x_grad) {
  auto input_dim = x_grad->dims();
  if (dim < 0) {
    dim += static_cast<int>(input_dim.size());
  }

  DenseTensor index;
  PADDLE_ENFORCE_EQ(repeats_tensor.dims()[0] == x_grad->dims()[dim],
                    true,
                    common::errors::InvalidArgument(
                        "The length of Input(RepeatsTensor) must be the "
                        "same as length of Input(X) in axis. "
                        "But received: [%s], required: [%d].",
                        repeats_tensor.dims()[0],
                        x_grad->dims()[dim]));

  const auto& index_type = repeats_tensor.dtype();

  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "Input(Repeats) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(index_type),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  auto output_dim = out_grad.dims();
  auto stride_dim = common::stride(input_dim);
  int64_t stride = stride_dim[dim];
  int64_t size = output_dim[dim];
  int64_t delta = input_dim[dim] - size;
  int64_t numel = x_grad->numel();
  int64_t out_nums = out_grad.numel();
  auto* out_grad_data = out_grad.data<T>();
  dev_ctx.template Alloc<T>(x_grad);
  auto* in_grad_data = x_grad->data<T>();
  auto stream = dev_ctx.stream();
  int vec_size = 8;
  vec_size = std::min(phi::GetVectorizedSize(in_grad_data), vec_size);
  auto config =
      phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel, vec_size);

  switch (vec_size) {
#define CASE_VEC_SIZE(__Sz)                                              \
  case __Sz:                                                             \
    index_select_grad_init<T, __Sz>                                      \
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>( \
            in_grad_data, numel);                                        \
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

  if (index_type == DataType::INT64) {
    phi::funcs::RepeatsTensor2IndexTensorFunctor<Context, int64_t>()(
        dev_ctx, repeats_tensor, &index);
    int64_t index_nums = index.numel();

    const int64_t* index_data = index.data<int64_t>();
    index_select_grad_cuda_kernel<T, int64_t>
        <<<(out_nums + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
           PADDLE_CUDA_NUM_THREADS,
           0,
           stream>>>(out_grad_data,
                     in_grad_data,
                     index_data,
                     out_nums,
                     stride,
                     size,
                     delta);
  } else {
    phi::funcs::RepeatsTensor2IndexTensorFunctor<Context, int>()(
        dev_ctx, repeats_tensor, &index);
    int64_t index_nums = index.numel();

    const int* index_data = index.data<int>();
    index_select_grad_cuda_kernel<T, int>
        <<<(out_nums + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
           PADDLE_CUDA_NUM_THREADS,
           0,
           stream>>>(out_grad_data,
                     in_grad_data,
                     index_data,
                     out_nums,
                     stride,
                     size,
                     delta);
  }
}

template <typename T, typename Context>
void RepeatInterleaveGradKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& out_grad,
                                int repeats,
                                int dim,
                                DenseTensor* x_grad) {
  if (x_grad && x_grad->numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  auto input_dim = x_grad->dims();
  auto output_grad_dim = out_grad.dims();

  const int ndim = input_dim.size();
  dim = (dim < 0) ? ndim + dim : dim;

  std::vector<int64_t> reshape_shape = vectorize(input_dim);
  reshape_shape.insert(reshape_shape.begin() + dim + 1, repeats);

  DenseTensor out_grad_copy;
  out_grad_copy.set_meta(out_grad.meta());
  out_grad_copy.ShareBufferWith(out_grad, true);

  out_grad_copy.Resize(make_ddim(reshape_shape));

  SumKernel<T, Context>(dev_ctx,
                        out_grad_copy,
                        phi::IntArray({dim + 1}),
                        x_grad->dtype(),
                        false,
                        x_grad);
}
}  // namespace phi

PD_REGISTER_KERNEL(repeat_interleave_with_tensor_index_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveWithTensorIndexGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(repeat_interleave_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
