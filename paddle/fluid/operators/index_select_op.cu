// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/index_select_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
__global__ void index_select_cuda_kernel(const T* input, T* output,
                                         int64_t* index, int64_t N,
                                         int64_t stride, int64_t size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int64_t dim_idx = idx % (stride * size) / stride;
  int64_t src_dim_idx = index[dim_idx];
  int64_t input_idx = idx + (src_dim_idx - dim_idx) * stride output[idx] =
                          input[input_idx];
}

template <typename T>
__global__ void index_select_cuda_kernel_int(const T* input, T* output,
                                             int* index, int64_t N,
                                             int64_t stride, int64_t size) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int64_t dim_idx = idx % (stride * size) / stride;
  int src_dim_idx = index[dim_idx];
  int64_t input_idx = idx + (src_dim_idx - dim_idx) * stride output[idx] =
                          input[input_idx];
}

template <typename DeviceContext, typename T>
class IndexSelectCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* index = context.Input<LoDTensor>("Index");
    auto* out = context.Output<LoDTensor>("Out");
    int dim = context.Attr<int>("dim");
    auto input_dim = in->dims();
    dim = dim >= 0 ? dim : dim + input_dim.size();
    auto stride_dim = framework::stride(input_dim) int64_t stride =
        stride_dim[dim];
    int64_t size = input_dim[dim];

    const auto& index_type = index->type();
    bool index_type_match = index_type == framework::proto::VarType::INT64 ||
                            index_type == framework::proto::VarType::INT32;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto* in_data = in->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int64_t numel = out_data->numel();

    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();

    if (index_type == framework::proto::VarType::INT64) {
      int64_t* index_data = index->data<int64_t>();
      index_select_cuda_kernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                                     PADDLE_CUDA_NUM_THREADS,
                                 PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          in_data, out_data, index_data, numel, stride, size);
    } else {
      int* index_data = index->data<int>();
      index_select_cuda_kernel_int<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                                         PADDLE_CUDA_NUM_THREADS,
                                     PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          in_data, out_data, index_data, numel, stride, size);
    }
  }
};

template <typename DeviceContext, typename T>
class IndexSelectGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    index_select,
    ops::IndexSelectCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSelectCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSelectCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSelectCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    index_select_grad,
    ops::IndexSelectGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSelectGradCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSelectGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSelectGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                   int64_t>);
