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

#include "paddle/fluid/operators/optimizers/dpsgd_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class DpsgdOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Param"), true,
                      "Input(Param) of DpsgdOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Grad"), true,
                      "Input(Grad) of DpsgdOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("LearningRate"), true,
                      "Input(LearningRate) of DpsgdOp should not be null.");
    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("Param").front(),
        framework::proto::VarType::LOD_TENSOR,
        "The input var's type should be LoDTensor, but the received is %s",
        ctx->Inputs("Param").front(), ctx->GetInputsVarType("Param").front());
    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("Grad").front(),
        framework::proto::VarType::LOD_TENSOR,
        "The input var's type should be LoDTensor, but the received is %s",
        ctx->Inputs("Grad").front(), ctx->GetInputsVarType("Grad").front());

    PADDLE_ENFORCE_EQ(ctx->HasOutput("ParamOut"), true,
                      "Output(ParamOut) of DpsgdOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "Learning rate should have 1 dimension");
    auto param_dims = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Grad"),
        "Param and Grad input of DpsgdOp should have same dimension");

    ctx->SetOutputDim("ParamOut", param_dims);
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("Param")->type(),
                                   ctx.GetPlace());
  }
};

class DpsgdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("LearningRate", "(Tensor) Learning rate");

    AddOutput("ParamOut", "(Tensor) Output parameter");

    AddAttr<float>("clip",
                   "(float, default 0.9) "
                   "Exponential decay rate for the "
                   "1st moment estimates.")
        .SetDefault(10.0f);
    AddAttr<float>("batch_size",
                   "(float, default 0.999) "
                   "exponential decay rate for the weighted "
                   "infinity norm estimates.")
        .SetDefault(16.0f);
    AddAttr<float>("sigma",
                   "(float, default 1.0e-8) "
                   "Constant for numerical stability")
        .SetDefault(1.0f);
    AddComment(R"DOC(
Dpsgd Optimizer.

We implement the Dpsgd optimizer according to CCS16 paper - 
Deep Learning with Differential Privacy.

Dpsgd updates:
CCS16 - Deep Learning with Differential Privacy.
[https://arxiv.org/abs/1607.00133]

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(dpsgd, ops::DpsgdOp, ops::DpsgdOpMaker);
REGISTER_OP_CPU_KERNEL(
    dpsgd, ops::DpsgdOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DpsgdOpKernel<paddle::platform::CPUDeviceContext, double>);
