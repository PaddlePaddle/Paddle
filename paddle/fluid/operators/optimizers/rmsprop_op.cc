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
    PADDLE_ENFORCE_EQ(ctx->HasInput("Param"), true,
                      platform::errors::NotFound(
                          "Input(Param) of RmspropOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("MeanSquare"), true,
        platform::errors::NotFound(
            "Input(MeanSquare) of RmspropOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("LearningRate"), true,
        platform::errors::NotFound(
            "Input(LearningRate) of RmspropOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Grad"), true,
                      platform::errors::NotFound(
                          "Input(Grad) of RmspropOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Moment"), true,
                      platform::errors::NotFound(
                          "Input(Moment) of RmspropOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->GetInputsVarType("Param").front(),
                      framework::proto::VarType::LOD_TENSOR,
                      platform::errors::InvalidArgument(
                          "The input var's type in RmspropOp should be "
                          "LoDTensor, but the received is %s",
                          ctx->GetInputsVarType("Param").front()));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("ParamOut"), true,
        platform::errors::NotFound(
            "Output(param_out) of RmspropOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("MomentOut"), true,
        platform::errors::NotFound(
            "Output(MomentOut) of RmspropOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("MeanSquareOut"), true,
        platform::errors::NotFound(
            "Output(MeanSquareOut) of RmspropOp should not be null."));
    if (ctx->Attrs().Get<bool>("centered")) {
      PADDLE_ENFORCE_EQ(
          ctx->HasOutput("MeanGradOut"), true,
          platform::errors::NotFound(
              "Output(MeanGradOut) of RmspropOp should not be null."));
    }

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(
        param_dim, ctx->GetInputDim("Grad"),
        platform::errors::InvalidArgument(
            "Param and grad input of RmspropOp should have the same dimension. "
            "But received Param's dim [%s] and Grad's dim [%s].",
            param_dim, ctx->GetInputDim("Grad")));
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("Moment"),
                      platform::errors::InvalidArgument(
                          "Param and Momentum input of RmspropOp "
                          "should have the same dimension. But received "
                          "Param's dim [%s] and Moment [%s]",
                          param_dim, ctx->GetInputDim("Moment")));
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("MeanSquare"),
                      platform::errors::InvalidArgument(
                          "Param and Momentum input of RmspropOp "
                          "should have the same dimension. But received "
                          "Param's dim [%s] and MeanSquare [%s]",
                          param_dim, ctx->GetInputDim("MeanSquare")));

    auto lr_dim = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(phi::product(lr_dim), 1,
                      platform::errors::InvalidArgument(
                          "Learning Rate of RmspropOp should be a scalar. But "
                          "received LearningRate's dim [%s]",
                          phi::product(lr_dim)));

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
    rmsprop, ops::RmspropOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::RmspropOpKernel<paddle::platform::CPUDeviceContext, double>);
