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

#include "paddle/fluid/operators/optimizers/gftrl_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class GFTRLOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of G-FTRL should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("SquaredAccumulator"),
                   "Input(SquaredAccumulator) of G-FTRL should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LinearAccumulator"),
                   "Input(LinearAccumulator) of G-FTRL should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of G-FTRL should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of G-FTRL should not be null.");
    PADDLE_ENFORCE(
        ctx->GetInputsVarType("Param").front() ==
            framework::proto::VarType::LOD_TENSOR,
        "The input var's type should be LoDTensor, but the received is %s",
        ctx->Inputs("Param").front(), ctx->GetInputsVarType("Param").front());

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of G-FTRL should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("SquaredAccumOut"),
                   "Output(SquaredAccumOut) of G-FTRL should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("LinearAccumOut"),
                   "Output(LinearAccumOut) of G-FTRL should not be null.");

    auto param_dim = ctx->GetInputDim("Param");
    PADDLE_ENFORCE_EQ(param_dim, ctx->GetInputDim("Grad"),
                      "Two input of G-FTRL Op's dimension must be same.");

    auto lr_dim = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_NE(framework::product(lr_dim), 0,
                      "Maybe the Input variable LearningRate has not "
                      "been initialized. You may need to confirm "
                      "if you put exe.run(startup_program) "
                      "after optimizer.minimize function.");
    PADDLE_ENFORCE_EQ(framework::product(lr_dim), 1,
                      "Learning Rate should be a scalar.");

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

class GFTRLOpMaker : public framework::OpProtoAndCheckerMaker {
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
G-FTRL (Group-Sparsity-Regularized FTRL) Operator.

Optimizer that implements the G-FTRL algorithm:

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

The paper that proposed Group-Sparsity-Regularized FTRL (G-FTRL):
(https://research.fb.com/wp-content/uploads/2019/09/Feature-Selection-for-Facebook-Feed-Ranking-System-via-a-Group-Sparsity-Regularized-Training-Algorithm.pdf)

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(gftrl, ops::GFTRLOp, ops::GFTRLOpMaker);
REGISTER_OP_CPU_KERNEL(
    gftrl, ops::GFTRLOpKernel<paddle::platform::CPUDeviceContext, float>);
