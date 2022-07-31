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
#include "paddle/phi/kernels/funcs/repeat_interleave_with_tensor_index_grad.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#endif

namespace phi {

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename IndexT>
__global__ void index_select_grad_cuda_kernel(const T* output_grad,
                                              T* input_grad,
                                              const IndexT* index,
                                              int64_t nums,
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
  paddle::platform::CudaAtomicAdd(&input_grad[input_idx], output_grad[idx]);
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
    const Context& dev_ctx,
    const DenseTensor& out_grad,
    const DenseTensor& repeats_tensor,
    int dim,
    DenseTensor* out_grad) {
  auto place = ctx.GetPlace();
  auto cpu_place = phi::CPUPlace();

  auto input_dims = x_grad.dims();
  if (dim < 0) {
    dim += input_dims.size();
  }

  DenseTensor index;
  PADDLE_ENFORCE_EQ(repeats_tensor.dims()[0] == x.dims()[dim],
                    true,
                    platform::errors::InvalidArgument(
                        "The length of Input(RepeatsTensor) must be the "
                        "same as length of Input(X) in axis. "
                        "But received: [%s], required: [%d].",
                        repeats_tensor.dims()[0],
                        x.dims()[dim]));

  const auto& index_type =
      framework::TransToProtoVarType(repeats_tensor->dtype());

  bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                          index_type == framework::proto::VarType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      platform::errors::InvalidArgument(
          "Input(Repeats) holds the wrong type, it holds %s, but "
          "desires to be %s or %s",
          paddle::framework::DataTypeToString(index_type),
          paddle::framework::DataTypeToString(framework::proto::VarType::INT32),
          paddle::framework::DataTypeToString(
              framework::proto::VarType::INT64)));
  if (place == cpu_place) {
    if (index_type == framework::proto::VarType::INT32) {
      RepeatsTensor2IndexTensor<int>(*repeats_tensor, &index);
      IndexSelectGradInner<Context, T, int>(
          context, *out_grad, index, x_grad, dim);
    } else if (index_type == framework::proto::VarType::INT64) {
      RepeatsTensor2IndexTensor<int64_t>(*repeats_tensor, &index);
      IndexSelectGradInner<Context, T, int64_t>(
          context, *out_grad, index, x_grad, dim);
    }
  }
#if defined(__NVCC__) || defined(__HIPCC__)
  else {
    auto output_dim = out_grad.dims();
    auto stride_dim = phi::stride(input_dim);
    int64_t stride = stride_dim[dim];
    int64_t size = output_dim[dim];
    int64_t delta = input_dim[dim] - size;
    int64_t numel = x_grad.numel();
    int64_t out_nums = out_grad->numel();
    auto* out_grad_data = out_grad->data<T>();
    ctx.template Alloc<T>(x_grad);
    auto* in_grad_data = x_grad->data<T>();
    auto stream = ctx.stream();
    index_select_grad_init<T>
        <<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
           PADDLE_CUDA_NUM_THREADS,
           0,
           stream>>>(in_grad_data, numel);

    int repeats = context.Attr<int>("Repeats");
    framework::LoDTensor index;
    if (context.HasInput("RepeatsTensor")) {
      auto repeats_tensor =
          context.Input<framework::LoDTensor>("RepeatsTensor");

      const auto& index_type =
          framework::TransToProtoVarType(repeats_tensor->dtype());
      bool index_type_match = index_type == framework::proto::VarType::INT64 ||
                              index_type == framework::proto::VarType::INT32;
      PADDLE_ENFORCE_EQ(
          index_type_match,
          true,
          platform::errors::InvalidArgument(
              "Input(Index) holds the wrong type, it holds %s, but "
              "desires to be %s or %s",
              paddle::framework::DataTypeToString(index_type),
              paddle::framework::DataTypeToString(
                  framework::proto::VarType::INT32),
              paddle::framework::DataTypeToString(
                  framework::proto::VarType::INT64)));

      if (index_type == framework::proto::VarType::INT64) {
        RepeatsTensor2IndexTensor<DeviceContext, int64_t>(*repeats_tensor,
                                                          &index);
        int64_t index_nums = index.numel();

        const int64_t* index_data = index.data<int64_t>();
        index_select_grad_cuda_kernel<T, int64_t>
            <<<(out_nums + PADDLE_CUDA_NUM_THREADS - 1) /
                   PADDLE_CUDA_NUM_THREADS,
               PADDLE_CUDA_NUM_THREADS,
               0,
               stream>>>(output_grad_data,
                         in_grad_data,
                         index_data,
                         index_nums,
                         out_nums,
                         stride,
                         size,
                         delta);
        platform::GpuStreamSync(stream);
      } else {
        RepeatsTensor2IndexTensor<DeviceContext, int>(*repeats_tensor, &index);
        int64_t index_nums = index.numel();

        const int* index_data = index.data<int>();
        index_select_grad_cuda_kernel<T, int>
            <<<(out_nums + PADDLE_CUDA_NUM_THREADS - 1) /
                   PADDLE_CUDA_NUM_THREADS,
               PADDLE_CUDA_NUM_THREADS,
               0,
               stream>>>(output_grad_data,
                         in_grad_data,
                         index_data,
                         index_nums,
                         out_nums,
                         stride,
                         size,
                         delta);
        platform::GpuStreamSync(stream);
      }
    }
#endif
  }
