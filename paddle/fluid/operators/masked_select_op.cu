/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>
#include "paddle/fluid/operators/masked_select_op.h"
#include "paddle/pten/kernels/masked_select_kernel.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class MaskedSelectCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto input = ctx.Input<framework::Tensor>("X");
    auto mask = ctx.Input<framework::Tensor>("Mask");
    auto out = ctx.Output<framework::Tensor>("Y");

    auto& dev_ctx = ctx.device_context<DeviceContext>();
    pten::MaskedSelectKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *input, *mask, out);
  }
};

template <typename DeviceContext, typename T>
class MaskedSelectGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto input = ctx.Input<framework::Tensor>(framework::GradVarName("Y"));
    auto mask = ctx.Input<framework::Tensor>("Mask");
    auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto& dev_ctx = ctx.device_context<DeviceContext>();
    pten::MaskedSelectGradKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *input, *x, *mask, out);
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    masked_select,
    ops::MaskedSelectCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MaskedSelectCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MaskedSelectCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MaskedSelectCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    masked_select_grad,
    ops::MaskedSelectGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MaskedSelectGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                    double>,
    ops::MaskedSelectGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::MaskedSelectGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                    int64_t>);
