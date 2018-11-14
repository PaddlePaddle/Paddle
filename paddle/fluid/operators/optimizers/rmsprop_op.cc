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

#include "paddle/fluid/operators/optimizers/rmsprop_op.h"

namespace paddle {
namespace operators {

class RmspropOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of RmspropOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("MeanSquare"),
                   "Input(MeanSquare) of RmspropOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of RmspropOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of RmspropOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Moment"),
                   "Input(Moment) of RmspropOp should not be null.");
    PADDLE_ENFORCE(
        ctx->GetInputsVarType("Param").front() ==
            framework::proto::VarType::LOD_TENSOR,
        "The input var's type should be LoDTensor, but the received is %s",
        ctx->Inputs("Param").front(), ctx->GetInputsVarType("Param").front());

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(param_out) of RmspropOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MomentOut"),
                   "Output(MomentOut) of RmspropOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MeanSquareOut"),
                   "Output(MeanSquareOut) of RmspropOp should not be null.");
    if (ctx->Attrs().Get<bool>("centered")) {
      PADDLE_ENFORCE(ctx->HasOutput("MeanGradOut"),
                     "Output(MeanGradOut) of RmspropOp should not be null.");
    }

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Grad"),
        "Param and grad input of RmspropOp should have the same dimension.");
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("Moment"),
                      "Param and Momentum input of RmspropOp "
                      "should have the same dimension.");
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("MeanSquare"),
                      "Param and Momentum input of RmspropOp "
                      "should have the same dimension.");

    auto lr_dim = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dim), 1,
                      "Learning Rate should be a scalar.");

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("MomentOut", param_dim);
    ctx->SetOutputDim("MeanSquareOut", param_dim);
    if (ctx->Attrs().Get<bool>("centered")) {
      ctx->SetOutputDim("MeanGradOut", param_dim);
    }
  }
};

class RmspropOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(Tensor, default Tensor<float>) "
             "Input parameter value that has to be updated.");
    AddInput("MeanSquare",
             "(Tensor, default Tensor<float>)"
             " The mean square value that gets updated.");
    AddInput("MeanGrad",
             "(Tensor, default Tensor<float>)"
             " The moving average of gradient")
        .AsDispensable();
    AddInput("LearningRate",
             "(Tensor, default Tensor<float>) "
             "The learning rate should be a tensor of size 1.");
    AddInput("Grad",
             "(Tensor, default Tensor<float>) "
             "Input gradient of the parameter.");
    AddInput("Moment",
             "(Tensor, default Tensor<float>) The moment that gets updated.");

    AddOutput("ParamOut", "(Tensor) Output updated parameter value.");
    AddOutput("MomentOut", "(Tensor) Output updated moment.");
    AddOutput("MeanSquareOut", "(Tensor) Output Mean squared updated value.");
    AddOutput("MeanGradOut",
              "(Tensor) Output moving average of gradient updated value.");

    AddAttr<float>("epsilon",
                   "(float, default 1e-10) Constant "
                   "for numerical stability.")
        .SetDefault(1.0e-10f);
    AddAttr<float>("decay",
                   "(float, default 0.9) "
                   "Discounting factor for coming gradient.")
        .SetDefault(0.9f);
    AddAttr<float>("momentum", "(float, default 0.0) Constant value.")
        .SetDefault(0.0f);
    AddAttr<bool>("centered", "(bool, default false) use centered rmsprop.")
        .SetDefault(false);
    AddComment(R"DOC(
Rmsprop Optimizer. 

$$
MeanSquareOut = decay * MeanSquare + (1 - decay) * Grad * Grad \\
MomentOut = momentum * Moment +
            \frac{LearningRate * Grad}{\sqrt{MeanSquareOut + epsilon}} \\
ParamOut = Param -  MomentOut
$$

if centered is true:

mean_grad = decay * mean_square{t-1} + (1-decay) * gradient
mean_square = decay * mean_square{t-1} + (1-decay) * gradient ** 2
mom = momentum * mom{t-1} + learning_rate * g_t /
    sqrt(mean_square - mean_grad**2 + epsilon)
param -= mom

The original slides that proposed Rmsprop: Slide 29 of
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(rmsprop, ops::RmspropOp, ops::RmspropOpMaker);
REGISTER_OP_CPU_KERNEL(
    rmsprop, ops::RmspropOpKernel<paddle::platform::CPUDeviceContext, float>);
