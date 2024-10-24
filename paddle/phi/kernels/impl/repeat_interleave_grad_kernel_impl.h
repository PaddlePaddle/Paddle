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

#pragma once

#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cpu/index_select_impl.h"
#include "paddle/phi/kernels/repeat_interleave_grad_kernel.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#ifdef __NVCC__
#include "cub/cub.cuh"
#else
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#endif

#include "paddle/phi/kernels/funcs/repeat_tensor2index_tensor.h"

namespace phi {

#if defined(__NVCC__) || defined(__HIPCC__)
using phi::PADDLE_CUDA_NUM_THREADS;

template <typename T, typename IndexT>
__global__ void index_select_grad_cuda_kernel(const T* output_grad,
                                              T* input_grad,
                                              const IndexT* index,
                                              int64_t N,
                                              int64_t stride,
                                              int64_t size,
                                              int64_t delta) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int64_t pre_idx = idx / (stride * size);
  int64_t dim_idx = idx % (stride * size) / stride;
  IndexT src_dim_idx = index[dim_idx];
  int64_t input_idx = idx + (delta * pre_idx + src_dim_idx - dim_idx) * stride;
  phi::CudaAtomicAdd(&input_grad[input_idx], output_grad[idx]);
}

template <typename T>
__global__ void index_select_grad_init(T* input_grad, int64_t N) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }
  input_grad[idx] = 0.0;
}
#endif
template <typename T, typename Context>
void RepeatInterleaveWithTensorIndexGradKernel(
    const Context& ctx,
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
#if defined(__NVCC__) || defined(__HIPCC__)

  auto output_dim = out_grad.dims();
  auto stride_dim = common::stride(input_dim);
  int64_t stride = stride_dim[dim];
  int64_t size = output_dim[dim];
  int64_t delta = input_dim[dim] - size;
  int64_t numel = x_grad->numel();
  int64_t out_nums = out_grad.numel();
  auto* out_grad_data = out_grad.data<T>();
  ctx.template Alloc<T>(x_grad);
  auto* in_grad_data = x_grad->data<T>();
  auto stream = ctx.stream();
  index_select_grad_init<T>
      <<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
         PADDLE_CUDA_NUM_THREADS,
         0,
         stream>>>(in_grad_data, numel);

  if (index_type == DataType::INT64) {
    phi::funcs::RepeatsTensor2IndexTensor<Context, int64_t>(
        ctx, repeats_tensor, &index);
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
    phi::funcs::RepeatsTensor2IndexTensor<Context, int>(
        ctx, repeats_tensor, &index);
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
#endif
}

template <typename T, typename Context>
void RepeatInterleaveGradKernel(const Context& ctx,
                                const DenseTensor& x,
                                const DenseTensor& out_grad,
                                int repeats,
                                int dim,
                                DenseTensor* x_grad) {
  auto input_dim = x_grad->dims();
  if (dim < 0) {
    dim += input_dim.size();
  }

  DenseTensor index;
#if defined(__NVCC__) || defined(__HIPCC__)
  auto output_dim = out_grad.dims();
  auto stride_dim = common::stride(input_dim);
  int64_t stride = stride_dim[dim];
  int64_t size = output_dim[dim];
  int64_t delta = input_dim[dim] - size;
  int64_t numel = x_grad->numel();
  int64_t out_nums = out_grad.numel();
  auto* out_grad_data = out_grad.data<T>();
  ctx.template Alloc<T>(x_grad);
  auto* in_grad_data = x_grad->data<T>();
  auto stream = ctx.stream();
  index_select_grad_init<T>
      <<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
         PADDLE_CUDA_NUM_THREADS,
         0,
         stream>>>(in_grad_data, numel);
  int64_t index_size = x_grad->dims()[dim] * repeats;
  std::vector<int> index_vec(index_size);
  for (int i = 0; i < x_grad->dims()[dim]; i++) {
    std::fill_n(index_vec.begin() + i * repeats, repeats, i);
  }
  index.Resize(common::make_ddim({index_size}));
  phi::TensorFromVector<int>(index_vec, ctx, &index);

  const int* index_data = index.data<int>();
  int64_t index_nums = index.numel();
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
#endif
}
}  // namespace phi
