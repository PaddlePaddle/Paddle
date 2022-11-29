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

#include "paddle/fluid/operators/optimizers/ftrl_op.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
class FTRLOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Param"), "Input", "Param", "FTRL");
    OP_INOUT_CHECK(ctx->HasInput("SquaredAccumulator"),
                   "Input",
                   "SquaredAccumulator",
                   "FTRL");
    OP_INOUT_CHECK(ctx->HasInput("LinearAccumulator"),
                   "Input",
                   "LinearAccumulator",
                   "FTRL");
    OP_INOUT_CHECK(ctx->HasInput("Grad"), "Input", "Grad", "FTRL");
    OP_INOUT_CHECK(
        ctx->HasInput("LearningRate"), "Input", "LearningRate", "FTRL");

    OP_INOUT_CHECK(ctx->HasOutput("ParamOut"), "Output", "ParamOut", "FTRL");
    OP_INOUT_CHECK(
        ctx->HasOutput("SquaredAccumOut"), "Output", "SquaredAccumOut", "FTRL");
    OP_INOUT_CHECK(
        ctx->HasOutput("LinearAccumOut"), "Output", "LinearAccumOut", "FTRL");

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(param_dim,
                      ctx->GetInputDim("Grad"),
                      platform::errors::InvalidArgument(
                          "Two input of FTRL Op's dimension must be same, but "
                          "param_dim is %d, Grad is %d",
                          param_dim,
                          ctx->GetInputDim("Grad")));

    auto lr_dim = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(phi::product(lr_dim),
                      0,
                      platform::errors::InvalidArgument(
                          "Maybe the Input variable LearningRate has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function."));
    PADDLE_ENFORCE_EQ(phi::product(lr_dim),
                      1,
                      platform::errors::InvalidArgument(
                          "Learning Rate should be a scalar, but got %d",
                          phi::product(lr_dim)));

    ctx->SetOutputDim("ParamOut", param_dim);
    ctx->SetOutputDim("SquaredAccumOut", param_dim);
    ctx->SetOutputDim("LinearAccumOut", param_dim);
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Param");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class FTRLOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param",
             "(Tensor, default Tensor<float>) "
             "Input parameter value that has to be updated.");
    AddInput("SquaredAccumulator",
             "(Tensor, default Tensor<float>) "
             "Accumulator that accumulates squared gradients.");
    AddInput("LinearAccumulator",
             "(Tensor, default Tensor<float>) "
             "Accumulator that accumulates linear gradients.");
    AddInput("Grad",
             "(Tensor, default Tensor<float>) "
             "Input gradient of the parameter.");
    AddInput("LearningRate",
             "(Tensor, default Tensor<float>) "
             "The learning rate should be a tensor of size 1.");

    AddOutput("ParamOut", "(Tensor) Output updated parameter value.");
    AddOutput("SquaredAccumOut",
              "(Tensor) Output accumulated squared"
              " gradients.");
    AddOutput("LinearAccumOut",
              "(Tensor) Output accumulated linear"
              " gradients.");

    AddAttr<float>("l1",
                   "(float, default 0.0) "
                   "L1 regularization strength.")
        .SetDefault(0.0f);
    AddAttr<float>("l2",
                   "(float, default 0.0) "
                   "L2 regularization strength.")
        .SetDefault(0.0f);
    AddAttr<float>("lr_power",
                   "(float, default -0.5f) "
                   "Learning Rate Power.")
        .SetDefault(-0.5f);
    AddComment(R"DOC(
FTRL (Follow The Regularized Leader) Operator.

Optimizer that implements the FTRL algorithm:

$$
new\_accum = squared\_accum + grad^2 \\
if (lr\_power == -0.5) {
   linear\_accum += grad - (\surd(new\_accum) - \surd(squared\_accum)) /
                   (learning\_rate * param) \\
} else {
   linear\_accum += grad -
                  (new\_accum^{-lr\_power} - accum^{-lr\_power}) /
                  (learning\_rate * param) \\
}

x = (l1 * sign(linear\_accum) - linear\_accum)
if (lr\_power == -0.5) {
   y = \frac{\surd(new\_accum)}{learning\_rate} + (2 * l2) \\
   pre\_shrink = \frac{x}{y} \\
   param = (abs(linear\_accum) > l1).select(pre\_shrink, 0.0) \\
} else {
   y = \frac{new\_accum^{-lr\_power}}{learning\_rate} + (2 * l2) \\
   pre\_shrink = \frac{x}{y} \\
   param = (abs(linear\_accum) > l1).select(pre\_shrink, 0.0) \\
}
squared\_accum += grad^2;
$$

The paper that proposed Follow The Regularized Leader (FTRL):
(https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(ftrl, ops::FTRLOp, ops::FTRLOpMaker);
REGISTER_OP_CPU_KERNEL(ftrl, ops::FTRLOpKernel<phi::CPUContext, float>);
