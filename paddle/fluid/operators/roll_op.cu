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
#include "paddle/fluid/operators/roll_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
__global__ void roll_cuda_kernel(const T* input, T* output, int64_t N,
                                 int64_t start, int64_t size, int64_t stride) {
  int64_t output_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_idx >= N) {
    return;
  }
  int64_t dim_idx = output_idx % (stride * size) / stride;
  int64_t input_idx = dim_idx >= (size - start)
                          ? (output_idx - ((size - start) * stride))
                          : (output_idx + (start * stride));
  output[output_idx] = input[input_idx];
}

template <typename DeviceContext, typename T>
class RollCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    std::vector<int64_t> shifts = context.Attr<std::vector<int64_t>>("shifts");
    std::vector<int64_t> dims = context.Attr<std::vector<int64_t>>("axis");

    auto* in_data = in->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int64_t numel = in->numel();
    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();

    size_t nums = shifts.size();
    auto input_dim = in->dims();
    auto stride_dim = framework::stride(input_dim);

    int64_t size, dim, start, stride;
    for (size_t i = 0; i < nums; i++) {
      dim = dims[i] >= 0 ? dims[i] : dims[i] + input_dim.size();
      size = input_dim[dim];
      start = (size - shifts[i]) % size;
      stride = stride_dim[dim];

      roll_cuda_kernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                             PADDLE_CUDA_NUM_THREADS,
                         PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          in_data, out_data, numel, start, size, stride);
    }
  }
};

template <typename DeviceContext, typename T>
class RollGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* out = context.Output<LoDTensor>(framework::GradVarName("X"));
    std::vector<int64_t> shifts = context.Attr<std::vector<int64_t>>("shifts");
    std::vector<int64_t> dims = context.Attr<std::vector<int64_t>>("axis");

    auto* in_data = in->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    int64_t numel = in->numel();
    auto stream =
        context.template device_context<platform::CUDADeviceContext>().stream();

    size_t nums = shifts.size();
    auto input_dim = in->dims();
    auto stride_dim = framework::stride(input_dim);

    int64_t size, dim, start, stride;
    for (size_t i = 0; i < nums; i++) {
      dim = dims[i] >= 0 ? dims[i] : dims[i] + input_dim.size();
      size = input_dim[dim];
      start = (size + shifts[i]) % size;
      stride = stride_dim[dim];

      roll_cuda_kernel<<<(numel + PADDLE_CUDA_NUM_THREADS - 1) /
                             PADDLE_CUDA_NUM_THREADS,
                         PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          in_data, out_data, numel, start, size, stride);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    roll, ops::RollCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::RollCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::RollCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::RollCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    roll_grad,
    ops::RollGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::RollGradCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::RollGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::RollGradCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
