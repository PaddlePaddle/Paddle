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

#include <memory>
#include <string>

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/optimizers/sgd_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SGDNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* learning_rate = ctx.Input<framework::LoDTensor>("LearningRate");
    auto* param_var = ctx.Input<framework::LoDTensor>("Param");
    auto* grad_var = ctx.Input<framework::LoDTensor>("Grad");
    auto* param_out = ctx.Output<framework::LoDTensor>("ParamOut");

    param_out->mutable_data<T>(ctx.GetPlace());

    auto runner =
        NpuOpRunner("ApplyGradientDescent",
                    {*param_var, *learning_rate, *grad_var}, {*param_out}, {});

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);

    // NOTE(zhiqiu): ApplyGradientDescent updates params inplace, so
    // if param and param_out is not same, we need to do copy.
    if (param_out->data<T>() != param_var->data<T>()) {
      framework::TensorCopy(
          *param_var, ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(), param_out);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    sgd, ops::SGDNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SGDNPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::SGDNPUKernel<paddle::platform::NPUDeviceContext,
                      paddle::platform::float16>);
