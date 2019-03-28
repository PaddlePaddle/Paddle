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

#include "paddle/fluid/operators/optimizers/adam_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
class AdamOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Param"),
                   "Input(Param) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Grad"),
                   "Input(Grad) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Moment1"),
                   "Input(Moment1) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Moment2"),
                   "Input(Moment2) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("LearningRate"),
                   "Input(LearningRate) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Beta1Pow"),
                   "Input(Beta1Pow) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Beta2Pow"),
                   "Input(Beta2Pow) of AdamOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("ParamOut"),
                   "Output(ParamOut) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Moment1Out"),
                   "Output(Moment1Out) of AdamOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Moment2Out"),
                   "Output(Moment2Out) of AdamOp should not be null.");

    auto lr_dims = ctx->GetInputDim("LearningRate");
    PADDLE_ENFORCE_EQ(framework::product(lr_dims), 1,
                      "Learning rate should have 1 dimension");
    auto beta1_pow_dims = ctx->GetInputDim("Beta1Pow");
    PADDLE_ENFORCE_EQ(framework::product(beta1_pow_dims), 1,
                      "Beta1 power accumulator should have 1 dimension");
    auto beta2_pow_dims = ctx->GetInputDim("Beta2Pow");
    PADDLE_ENFORCE_EQ(framework::product(beta2_pow_dims), 1,
                      "Beta2 power accumulator should have 1 dimension");

    auto param_dims = ctx->GetInputDim("Param");
    if (ctx->GetInputsVarType("Grad")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(
          param_dims, ctx->GetInputDim("Grad"),
          "Param and Grad input of AdamOp should have same dimension");
    }
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment1"),
        "Param and Moment1 input of AdamOp should have same dimension");
    PADDLE_ENFORCE_EQ(
        param_dims, ctx->GetInputDim("Moment2"),
        "Param and Moment2 input of AdamOp should have same dimension");

    ctx->SetOutputDim("ParamOut", param_dims);
    ctx->SetOutputDim("Moment1Out", param_dims);
    ctx->SetOutputDim("Moment2Out", param_dims);
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = ctx.Input<Tensor>("Param")->type();
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class AdamOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("LearningRate", "(Tensor) Learning rate");
    AddInput("Moment1", "(Tensor) Input first moment");
    AddInput("Moment2", "(Tensor) Input second moment");
    AddInput("Beta1Pow", "(Tensor) Input beta1 power accumulator");
    AddInput("Beta2Pow", "(Tensor) Input beta2 power accumulator");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("Moment1Out", "(Tensor) Output first moment");
    AddOutput("Moment2Out", "(Tensor) Output second moment");

    AddAttr<float>("beta1",
                   "(float, default 0.9) "
                   "Exponential decay rate for the "
                   "first moment estimates.")
        .SetDefault(0.9f);
    AddAttr<float>("beta2",
                   "(float, default 0.999) "
                   "exponential decay rate for the "
                   "second moment estimates.")
        .SetDefault(0.999f);
    AddAttr<float>("epsilon",
                   "(float, default 1.0e-8) "
                   "Constant for numerical stability")
        .SetDefault(1.0e-8f);
    AddAttr<bool>(
        "lazy_mode",
        "(bool, default false) "
        "only update the parameter that has gradient in sparse update")
        .SetDefault(false);
    AddAttr<int64_t>("min_row_size_to_use_multithread",
                     "(int64_t, default 0) "
                     "when not zero, if param row size is larger then "
                     "min_row_size_to_use_multithread and "
                     "inner_op_parallelism is larger then 0, sparse update "
                     "will run in multithread mode")
        .SetDefault(1000);

    AddComment(R"DOC(
Adam Optimizer.

This implements the Adam optimizer from Section 2 of the Adam
paper : https://arxiv.org/abs/1412.6980.
Adam is a first-order gradient-based optimization method based on
adaptive estimates of lower-order moments.

Adam updates:

$$
moment\_1\_out = \beta_1 * moment\_1 + (1 - \beta_1) * grad \\
moment\_2_\out = \beta_2 * moment\_2 + (1 - \beta_2) * grad * grad \\
learning\_rate = learning\_rate *
                  \frac{\sqrt{1 - \beta_{2\_pow}}}{1 - \beta_{1\_pow}} \\
param\_out = param - learning\_rate * \frac{moment\_1}{\sqrt{moment\_2} + \epsilon}
$$

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(adam, ops::AdamOp, ops::AdamOpMaker);
REGISTER_OP_CPU_KERNEL(
    adam, ops::AdamOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AdamOpKernel<paddle::platform::CPUDeviceContext, double>);
