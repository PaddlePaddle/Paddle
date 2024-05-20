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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/cpu/index_select_impl.h"
#include "paddle/phi/kernels/repeat_interleave_kernel.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/backends/gpu/gpu_resources.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#endif

#include "paddle/phi/kernels/funcs/repeat_tensor2index_tensor.h"

namespace phi {

#if defined(__NVCC__) || defined(__HIPCC__)
using phi::PADDLE_CUDA_NUM_THREADS;
template <typename T, typename IndexT>
__global__ void index_select_cuda_kernel(const T* input,
                                         T* output,
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
  output[idx] = input[input_idx];
}
#endif

template <typename T, typename Context>
void RepeatInterleaveKernel(const Context& ctx,
                            const DenseTensor& x,
                            int repeats,
                            int dim,
                            DenseTensor* out) {
  PADDLE_ENFORCE_GT(repeats,
                    0,
                    phi::errors::InvalidArgument(
                        "repeats must grater than 0, but got %d", repeats));

  auto place = ctx.GetPlace();
  auto cpu_place = phi::CPUPlace();

  auto input_dim = x.dims();
  if (dim < 0) {
    dim += input_dim.size();
  }

  DenseTensor index;
  int64_t index_size = input_dim[dim] * repeats;
  std::vector<int> index_vec(index_size);
  for (int i = 0; i < input_dim[dim]; i++) {
    std::fill_n(index_vec.begin() + i * repeats, repeats, i);
  }
  index.Resize(common::make_ddim({index_size}));
  if (place == cpu_place) {
    DenseTensor x_copy = x;
    phi::TensorFromVector<int>(index_vec, ctx, &index);

    auto output_dim = common::vectorize(x.dims());
    output_dim[dim] = index_size;
    out->Resize(common::make_ddim(output_dim));
    phi::IndexSelectInner<Context, T, int>(ctx, &x_copy, index, out, dim);
#if defined(__NVCC__) || defined(__HIPCC__)
  } else {
    auto stride_dim = common::stride(input_dim);
    int64_t stride = stride_dim[dim];
    phi::TensorFromVector<int>(index_vec, ctx, &index);
    auto stream = ctx.stream();
    auto output_dim = common::vectorize(x.dims());
    output_dim[dim] = index_size;
    out->Resize(common::make_ddim(output_dim));
    ctx.template Alloc<T>(out);
    auto* out_data = out->data<T>();
    int64_t numel = out->numel();
    int64_t size = output_dim[dim];
    int64_t delta = input_dim[dim] - size;

    const int* index_data = index.data<int>();
    index_select_cuda_kernel<T, int>
        <<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
           PADDLE_CUDA_NUM_THREADS,
           0,
           stream>>>(
            x.data<T>(), out_data, index_data, numel, stride, size, delta);
  }
#else
  }
#endif
}

template <typename T, typename Context>
void RepeatInterleaveWithTensorIndexKernel(const Context& ctx,
                                           const DenseTensor& x,
                                           const DenseTensor& repeats_tensor,
                                           int dim,
                                           DenseTensor* out) {
  auto place = ctx.GetPlace();
  auto cpu_place = phi::CPUPlace();

  auto input_dim = x.dims();
  if (dim < 0) {
    dim += input_dim.size();
  }
  DenseTensor index;
  PADDLE_ENFORCE_EQ(repeats_tensor.dims()[0] == x.dims()[dim],
                    true,
                    phi::errors::InvalidArgument(
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
      phi::errors::InvalidArgument(
          "Input(RepeatsTensor) holds the wrong type, it holds %s, but "
          "desires to be %s or %s",
          DataTypeToString(index_type),
          DataTypeToString(phi::DataType::INT32),
          DataTypeToString(phi::DataType::INT64)));
  if (place == cpu_place) {
    auto x_copy = x;
    if (index_type == phi::DataType::INT32) {
      phi::funcs::RepeatsTensor2IndexTensor<Context, int>(
          ctx, repeats_tensor, &index);
      auto output_dim = common::vectorize(x.dims());
      output_dim[dim] = index.dims()[0];
      out->Resize(common::make_ddim(output_dim));
      IndexSelectInner<Context, T, int>(ctx, &x_copy, index, out, dim);
    } else if (index_type == phi::DataType::INT64) {
      phi::funcs::RepeatsTensor2IndexTensor<Context, int64_t>(
          ctx, repeats_tensor, &index);
      auto output_dim = common::vectorize(x.dims());
      output_dim[dim] = index.dims()[0];
      out->Resize(common::make_ddim(output_dim));
      IndexSelectInner<Context, T, int64_t>(ctx, &x_copy, index, out, dim);
    }
#if defined(__NVCC__) || defined(__HIPCC__)
  } else {
    auto stride_dim = common::stride(input_dim);
    int64_t stride = stride_dim[dim];
    auto stream = ctx.stream();
    auto* in_data = x.data<T>();
    if (index_type == phi::DataType::INT64) {
      phi::funcs::RepeatsTensor2IndexTensor<Context, int64_t>(
          ctx, repeats_tensor, &index);

      const int64_t* index_data = index.data<int64_t>();
      auto output_dim = common::vectorize(x.dims());
      output_dim[dim] = index.dims()[0];
      out->Resize(common::make_ddim(output_dim));
      T* out_data = ctx.template Alloc<T>(out);
      int64_t numel = out->numel();
      int64_t size = output_dim[dim];
      int64_t delta = input_dim[dim] - size;

      index_select_cuda_kernel<T, int64_t>
          <<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
             PADDLE_CUDA_NUM_THREADS,
             0,
             stream>>>(
              in_data, out_data, index_data, numel, stride, size, delta);
    } else {
      phi::funcs::RepeatsTensor2IndexTensor<Context, int>(
          ctx, repeats_tensor, &index);

      const int* index_data = index.data<int>();
      auto output_dim = common::vectorize(x.dims());
      output_dim[dim] = index.dims()[0];
      out->Resize(common::make_ddim(output_dim));
      T* out_data = ctx.template Alloc<T>(out);
      int64_t numel = out->numel();
      int64_t size = output_dim[dim];
      int64_t delta = input_dim[dim] - size;
      index_select_cuda_kernel<T, int>
          <<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
             PADDLE_CUDA_NUM_THREADS,
             0,
             stream>>>(
              in_data, out_data, index_data, numel, stride, size, delta);
    }
  }
#else
  }
#endif
}

}  // namespace phi
