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

#include "paddle/fluid/operators/momentum_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class MomentumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(param) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(grad) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Velocity"),
                   "Input(velocity) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of Momentum should not be null.");
    PADDLE_ENFORCE(
        ctx->GetInputsVarType("Param").front() ==
            framework::proto::VarType::LOD_TENSOR,
        "The input var's type should be LoDTensor, but the received is %s",
        ctx->Inputs("Param").front(), ctx->GetInputsVarType("Param").front());

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of Momentum should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("VelocityOut"),
                   "Output(VelocityOut) of Momentum should not be null.");

    auto param_dim = ctx->GetInputDim("Param");
    if (ctx->GetInputsVarType("Grad")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          param_dim, ctx->GetInputDim("Grad"),
          "Param and Grad input of MomentumOp should have the same dimension.");
      PADDLE_ENFORCE_EQ(
          param_dim, ctx->GetInputDim("Velocity"),
          "Param and Velocity of MomentumOp should have the same dimension.");
    }
    PADDLE_ENFORCE_EQ(framework::product(ctx->GetInputDim("LearningRate")), 1,
                      "Learning_rate should be a scalar");

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("VelocityOut", param_dim);
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = framework::GetDataTypeOfVar(ctx.InputVar("Param"));
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class MomentumOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    auto input_var = op_desc.Input("Param")[0];
    for (auto& out_var : op_desc.Output("ParamOut")) {
      if (block->FindRecursiveOrCreateVar(input_var).GetType() ==
          framework::proto::VarType::SELECTED_ROWS) {
        block->FindRecursiveOrCreateVar(out_var).SetType(
            framework::proto::VarType::SELECTED_ROWS);
      } else if (block->FindRecursiveOrCreateVar(input_var).GetType() ==
                 framework::proto::VarType::LOD_TENSOR) {
        block->FindRecursiveOrCreateVar(out_var).SetType(
            framework::proto::VarType::LOD_TENSOR);
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
