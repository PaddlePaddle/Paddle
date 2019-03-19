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

#include "paddle/fluid/operators/optimizers/momentum_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MomentumOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto& input_var = ctx->Input("Param")[0];
    for (auto& out_var : ctx->Output("ParamOut")) {
      if (ctx->GetType(input_var) == framework::proto::VarType::SELECTED_ROWS) {
        ctx->SetType(out_var, framework::proto::VarType::SELECTED_ROWS);
      } else if (ctx->GetType(input_var) ==
                 framework::proto::VarType::LOD_TENSOR) {
        ctx->SetType(out_var, framework::proto::VarType::LOD_TENSOR);
      } else {
        PADDLE_THROW(
            "Only support LodTensor and SelectedRows, Unexpected Input Type.");
      }
    }
  }
};

class MomentumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(Tensor, default Tensor<float>) "
             "Input parameter that has to be updated");
    AddInput("Grad",
             "(Tensor, default Tensor<float>) "
             "Input gradient of the parameter");
    AddInput("Velocity",
             "(Tensor, default Tensor<float>) "
             "Input velocity (corresponding to the parameter) "
             "that has to be updated");
    AddInput("LearningRate",
             "(Tensor, default Tensor<float>) "
             "Input learning rate");

    AddOutput("ParamOut",
              "(Tensor) This output is updated parameter. "
              "It shared memory with Input(Param).");
    AddOutput("VelocityOut",
              "(Tensor) This output is updated velocity. "
              "It shared memory with Input(Velocity).");

    AddAttr<float>("mu", "(float) Momentum coefficient");
    AddAttr<bool>("use_nesterov",
                  "(bool, default false) "
                  "Use Nesterov Momentum")
        .SetDefault(false);
    AddComment(R"DOC(
Momentum Optimizer.

This optimizer has a flag for Nestrov Momentum.
The update equations are as follows:

$$
velocity = mu * velocity + gradient \\
if (use\_nesterov):   \\
  param = param - (gradient + mu * velocity) * learning\_rate \\
else:   \\
  param = param - learning\_rate * velocity. \\
$$

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(momentum, ops::MomentumOp, ops::MomentumOpMaker,
                  paddle::framework::EmptyGradOpMaker,
                  ops::MomentumOpInferVarType);
REGISTER_OP_CPU_KERNEL(
    momentum, ops::MomentumOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MomentumOpKernel<paddle::platform::CPUDeviceContext, double>);
