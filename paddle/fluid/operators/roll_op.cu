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
#include "paddle/fluid/framework/array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/roll_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T, size_t N>
__global__ void RollCudaKernel(const T* input, T* output, int64_t Num,
                               paddle::framework::Array<int64_t, N> shifts,
                               paddle::framework::Array<int64_t, N> strides,
                               paddle::framework::Array<int64_t, N> sizes) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Num) {
    return;
  }

  int64_t output_idx = idx;
  int64_t dim_idx, dim_idx_shift;

#pragma unroll N
  for (size_t i = 0; i < N; i++) {
    dim_idx = (idx / strides[i]) % sizes[i];
    dim_idx_shift = (dim_idx + shifts[i]) % sizes[i];
    output_idx = output_idx + (dim_idx_shift - dim_idx) * strides[i];
  }
  output[output_idx] = input[idx];
}

template <typename T>
class RollKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
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

    std::vector<int64_t> strides(nums), sizes(nums);
    if (dims.size() == 0) {
      strides[0] = 1;
      sizes[0] = numel;
      shifts[0] = (shifts[0] % numel + numel) % numel;
    } else {
      for (size_t i = 0; i < nums; i++) {
        int dim = dims[i] >= 0 ? dims[i] : dims[i] + input_dim.size();
        int64_t size = input_dim[dim];

        shifts[i] = (shifts[i] % size + size) % size;
        strides[i] = stride_dim[dim];
        sizes[i] = size;
      }
    }

#define CallRollCudaKernel(N)                                                  \
  case N: {                                                                    \
    paddle::framework::Array<int64_t, N> _strides;                             \
    paddle::framework::Array<int64_t, N> _shifts;                              \
    paddle::framework::Array<int64_t, N> _sizes;                               \
    for (size_t idx = 0; idx < N; ++idx) {                                     \
      _strides[idx] = strides[idx];                                            \
      _shifts[idx] = shifts[idx];                                              \
      _sizes[idx] = sizes[idx];                                                \
    }                                                                          \
    RollCudaKernel<                                                            \
        T,                                                                     \
        N><<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,  \
             PADDLE_CUDA_NUM_THREADS, 0, stream>>>(in_data, out_data, numel,   \
                                                   _shifts, _strides, _sizes); \
    break;                                                                     \
  }

    switch (nums) {
      CallRollCudaKernel(1);
      CallRollCudaKernel(2);
      CallRollCudaKernel(3);
      CallRollCudaKernel(4);
      CallRollCudaKernel(5);
      CallRollCudaKernel(6);
      CallRollCudaKernel(7);
      CallRollCudaKernel(8);
      CallRollCudaKernel(9);
      default:
        break;
    }
  }
};

template <typename T>
class RollGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
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

    std::vector<int64_t> strides(nums), sizes(nums);
    if (dims.size() == 0) {
      strides[0] = 1;
      sizes[0] = numel;
      shifts[0] = ((-shifts[0]) % numel + numel) % numel;
    } else {
      for (size_t i = 0; i < nums; i++) {
        int dim = dims[i] >= 0 ? dims[i] : dims[i] + input_dim.size();
        int64_t size = input_dim[dim];

        shifts[i] = ((-shifts[i]) % size + size) % size;
        strides[i] = stride_dim[dim];
        sizes[i] = size;
      }
    }

    switch (nums) {
      CallRollCudaKernel(1);
      CallRollCudaKernel(2);
      CallRollCudaKernel(3);
      CallRollCudaKernel(4);
      CallRollCudaKernel(5);
      CallRollCudaKernel(6);
      CallRollCudaKernel(7);
      CallRollCudaKernel(8);
      CallRollCudaKernel(9);
      default:
        break;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    roll, ops::RollKernel<paddle::platform::CUDADeviceContext, float>,
    ops::RollKernel<paddle::platform::CUDADeviceContext, double>,
    ops::RollKernel<paddle::platform::CUDADeviceContext, int>,
    ops::RollKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    roll_grad, ops::RollGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::RollGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::RollGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::RollGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
