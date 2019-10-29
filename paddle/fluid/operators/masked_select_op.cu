/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <vector>
#include "paddle/fluid/operators/masked_select_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
__global__ void MaskedSelect(const int nums, const T* input_data,
                             const bool* mask_data, T* output_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  int j = index;
  for (size_t i = index; i < nums; i += offset) {
    if (mask_data[i]) {
      output_data[j] = input_data[i];
      j += offset;
    }
  }
}

template <typename DeviceContext, typename T>
class MaskedSelectOPCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::Tensor>("input");
    auto* mask = ctx.Input<framework::Tensor>("mask");
    auto* output = ctx.Output<framework::Tensor>("Out");

    const T* input_data = input->data<T>();
    auto* mask_data = mask->data<bool>();

    int out_dim = 0;
    for (size_t i = 0; i < mask->numel(); i++) {
      if (mask_data[i]) {
        out_dim++;
      }
    }
    output->Resize({out_dim});

    T* output_data = output->mutable_data<T>(ctx.GetPlace());

    int blocks = NumBlocks(input->numel());
    int threads = kNumCUDAThreads;

    MaskedSelect<T><<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>(
        input->numel(), input_data, mask_data, output_data);
  }
};

template <typename T>
__global__ void MaskedSelectGrad(const int nums, const T* output_grad_data,
                                 const bool* mask_data, T* input_grad_data) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  int j = index;
  for (size_t i = index; i < nums; i += offset) {
    if (mask_data[i]) {
      input_grad_data[i] = output_grad_data[j];
      j += offset;
    } else {
      input_grad_data[i] = 0;
    }
  }
}

template <typename DeviceContext, typename T>
class MaskedSelectGradOPCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* output_grad =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* input_grad =
        ctx.Output<framework::Tensor>(framework::GradVarName("input"));
    auto* mask = ctx.Input<framework::Tensor>("mask");

    // input_grad->Resize({mask->dims()})

    int blocks = NumBlocks(mask->numel());
    int threads = kNumCUDAThreads;

    auto* mask_data = mask->data<bool>();
    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());

    MaskedSelectGrad<
        T><<<blocks, threads, 0, ctx.cuda_device_context().stream()>>>(
        blocks, output_grad_data, mask_data, input_grad_data);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    masked_select,
    ops::MaskedSelectOPCUDAKernel<plat::CUDADeviceContext, float>,
    ops::MaskedSelectOPCUDAKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    masked_select_grad,
    ops::MaskedSelectGradOPCUDAKernel<plat::CUDADeviceContext, float>,
    ops::MaskedSelectGradOPCUDAKernel<plat::CUDADeviceContext, double>);
