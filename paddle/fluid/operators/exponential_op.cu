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

#include "paddle/fluid/operators/exponential_op.h"

namespace paddle {
namespace operators {

template <typename T>
class ExponentialKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    framework::Tensor* out = ctx.Output<framework::Tensor>("Out");
    auto& dev_cxt = ctx.template device_context<platform::CUDADeviceContext>();
    T lambda = static_cast<T>(ctx.Attr<float>("lambda"));

    distribution::uniform_distribution<T> dist;
    distribution::exponential_transform<T> trans(lambda);
    distribution::distribution_and_transform<T>(dev_cxt, out, dist, trans);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    exponential, ops::ExponentialKernel<plat::CUDADeviceContext, float>,
    ops::ExponentialKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    exponential_grad,
    ops::ExponentialGradKernel<plat::CUDADeviceContext, float>,
    ops::ExponentialGradKernel<plat::CUDADeviceContext, double>);
