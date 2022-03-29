/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/uniform_random_op.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace paddle {
namespace operators {
template <typename T>
class GPUUniformRandomInplaceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* tensor = context.Output<framework::Tensor>("Out");
    UniformRandom<T>(context, tensor);
  }
};

template <typename T>
class GPUUniformRandomInplaceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto dims = vectorize(dx->dims());
    const auto& dev_cxt =
        ctx.template device_context<platform::CUDADeviceContext>();
    float value = static_cast<float>(0.0f);
    phi::FullKernel<T>(
        static_cast<const typename paddle::framework::ConvertToPhiContext<
            paddle::platform::CUDADeviceContext>::TYPE&>(dev_cxt),
        dims, value, phi::DataType::UNDEFINED, dx);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    uniform_random_inplace,
    paddle::operators::GPUUniformRandomInplaceKernel<float>,
    paddle::operators::GPUUniformRandomInplaceKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    uniform_random_inplace_grad,
    paddle::operators::GPUUniformRandomInplaceGradKernel<float>,
    paddle::operators::GPUUniformRandomInplaceGradKernel<double>);
