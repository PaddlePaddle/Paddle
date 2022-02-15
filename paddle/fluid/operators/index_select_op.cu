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
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T, typename IndexT>
__global__ void index_select_cuda_kernel(const T* input, T* output,
                                         const IndexT* index, int64_t N,
                                         int64_t stride, int64_t size,
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

template <typename T, typename IndexT>
__global__ void index_select_grad_cuda_kernel(const T* output_grad,
                                              T* input_grad,
                                              const IndexT* index, int64_t nums,
                                              int64_t N, int64_t stride,
                                              int64_t size, int64_t delta) {
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

template <typename DeviceContext, typename T>
class IndexSelectCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* index = context.Input<LoDTensor>("Index");
    auto* out = context.Output<LoDTensor>("Out");
    int dim = context.Attr<int>("dim");
    auto input_dim = in->dims();
    auto output_dim = out->dims();
    dim = dim >= 0 ? dim : dim + input_dim.size();
    auto stride_dim = framework::stride(input_dim);
    int64_t stride = stride_dim[dim];
    int64_t size = output_dim[dim];
    int64_t delta = input_dim[dim] - size;

    const auto& index_type = framework::TransToProtoVarType(index->dtype());
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
    int64_t numel = out->numel();

    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();

    if (index_type == framework::proto::VarType::INT64) {
      const int64_t* index_data = index->data<int64_t>();
      index_select_cuda_kernel<T, int64_t><<<
          (numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
          PADDLE_CUDA_NUM_THREADS, 0, stream>>>(in_data, out_data, index_data,
                                                numel, stride, size, delta);
      platform::GpuStreamSync(stream);
    } else {
      const int* index_data = index->data<int>();
      index_select_cuda_kernel<T, int><<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                                             PADDLE_CUDA_NUM_THREADS,
                                         PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          in_data, out_data, index_data, numel, stride, size, delta);
      platform::GpuStreamSync(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class IndexSelectGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* output_grad = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* in_grad = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto* index = context.Input<LoDTensor>("Index");

    auto* output_grad_data = output_grad->data<T>();
    auto* in_grad_data = in_grad->mutable_data<T>(context.GetPlace());

    int dim = context.Attr<int>("dim");
    auto input_dim = in_grad->dims();
    auto output_dim = output_grad->dims();
    dim = dim >= 0 ? dim : dim + input_dim.size();
    auto stride_dim = framework::stride(input_dim);
    int64_t stride = stride_dim[dim];
    int64_t size = output_dim[dim];
    int64_t delta = input_dim[dim] - size;

    const auto& index_type = framework::TransToProtoVarType(index->dtype());
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

    int64_t numel = in_grad->numel();
    int64_t index_nums = index->numel();
    int64_t out_nums = output_grad->numel();

    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();

    index_select_grad_init<
        T><<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
             PADDLE_CUDA_NUM_THREADS, 0, stream>>>(in_grad_data, numel);

    if (index_type == framework::proto::VarType::INT64) {
      const int64_t* index_data = index->data<int64_t>();
      index_select_grad_cuda_kernel<T, int64_t><<<
          (out_nums + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
          PADDLE_CUDA_NUM_THREADS, 0, stream>>>(output_grad_data, in_grad_data,
                                                index_data, index_nums,
                                                out_nums, stride, size, delta);
      platform::GpuStreamSync(stream);
    } else {
      const int* index_data = index->data<int>();
      index_select_grad_cuda_kernel<T, int><<<
          (out_nums + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
          PADDLE_CUDA_NUM_THREADS, 0, stream>>>(output_grad_data, in_grad_data,
                                                index_data, index_nums,
                                                out_nums, stride, size, delta);
      platform::GpuStreamSync(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    index_select,
    ops::IndexSelectCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSelectCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSelectCUDAKernel<paddle::platform::CUDADeviceContext,
                               paddle::platform::float16>,
    ops::IndexSelectCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSelectCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    index_select_grad,
    ops::IndexSelectGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSelectGradCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSelectGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                   paddle::platform::float16>,
    ops::IndexSelectGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSelectGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                   int64_t>);
