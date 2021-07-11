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
#include <string>
#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/optimizers/sgd_op.h"

namespace paddle {
namespace operators {

template <typename T>
class NPUMomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();

    auto param = ctx.Input<framework::Tensor>("Param");
    auto velocity = ctx.Input<framework::Tensor>("Velocity");
    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    auto grad = ctx.Input<framework::Tensor>("Grad");
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
    param_out->mutable_data<T>(ctx.GetPlace());
    velocity_out->mutable_data<T>(ctx.GetPlace());

    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    Tensor mu_tensor;
    mu_tensor.mutable_data<T>(framework::make_ddim({1}), ctx.GetPlace());
    FillNpuTensorWithConstant<T>(&mu_tensor, mu);
    framework::TensorCopy(*param, ctx.GetPlace(), dev_ctx, param_out);
    framework::TensorCopy(*velocity, ctx.GetPlace(), dev_ctx, velocity_out);
    const auto& runner = NpuOpRunner(
        "ApplyMomentum",
        {*param_out, *velocity_out, *learning_rate, *grad, mu_tensor},
        {*param_out}, {{"use_nesterov", use_nesterov}});
    auto stream = dev_ctx.stream();
    runner.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(momentum, ops::NPUMomentumOpKernel<float>);
