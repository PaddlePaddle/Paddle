/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/optimizers/amsgrad_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

void AmsgradOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("Param"), true,
                    "Input(Param) of AmsgradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Grad"), true,
                    "Input(Grad) of AmsgradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Moment1"), true,
                    "Input(Moment1) of AmsgradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Moment2"), true,
                    "Input(Moment2) of AmsgradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("MaxMoment2"), true,
                    "Input(MaxMoment2) of AmsgradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("LearningRate"), true,
                    "Input(LearningRate) of AmsgradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Beta1Pow"), true,
                    "Input(Beta1Pow) of AmsgradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Beta2Pow"), true,
                    "Input(Beta2Pow) of AmsgradOp should not be null.");

  PADDLE_ENFORCE_EQ(ctx->HasOutput("ParamOut"), true,
                    "Output(ParamOut) of AmsgradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasOutput("Moment1Out"), true,
                    "Output(Moment1Out) of AmsgradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasOutput("Moment2Out"), true,
                    "Output(Moment2Out) of AmsgradOp should not be null.");

  PADDLE_ENFORCE_EQ(ctx->HasOutput("MaxMoment2Out"), true,
                    "Output(MaxMoment2Out) of AmsgradOp should not be null.");
  auto lr_dims = ctx->GetInputDim("LearningRate");
  PADDLE_ENFORCE_NE(framework::product(lr_dims), 0,
                    "Maybe the Input variable LearningRate has not "
                    "been initialized. You may need to confirm "
                    "if you put exe.run(startup_program) "
                    "after optimizer.minimize function.");
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
        "Param and Grad input of AmsgradOp should have same dimension");
  }
  PADDLE_ENFORCE_EQ(
      param_dims, ctx->GetInputDim("Moment1"),
      "Param and Moment1 input of AmsgradOp should have same dimension");
  PADDLE_ENFORCE_EQ(
      param_dims, ctx->GetInputDim("Moment2"),
      "Param and Moment2 input of AmsgradOp should have same dimension");
  PADDLE_ENFORCE_EQ(
      param_dims, ctx->GetInputDim("MaxMoment2"),
      "Param and MaxMoment2 input of AmsgradOp should have same dimension");
  ctx->SetOutputDim("ParamOut", param_dims);
  ctx->SetOutputDim("Moment1Out", param_dims);
  ctx->SetOutputDim("Moment2Out", param_dims);
  ctx->SetOutputDim("MaxMoment2Out", param_dims);
}

framework::OpKernelType AmsgradOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Param");
  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

class AmsgradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("Grad", "(Tensor) Input gradient");
    AddInput("LearningRate", "(Tensor) Learning rate");
    AddInput("Moment1", "(Tensor) Input first moment");
    AddInput("Moment2", "(Tensor) Input second moment");
    AddInput("MaxMoment2", "(Tensor) Input max second moment");

    AddInput("Beta1Pow", "(Tensor) Input beta1 power accumulator");
    AddInput("Beta2Pow", "(Tensor) Input beta2 power accumulator");

    AddOutput("ParamOut", "(Tensor) Output parameter");
    AddOutput("Moment1Out", "(Tensor) Output first moment");
    AddOutput("Moment2Out", "(Tensor) Output second moment");
    AddOutput("MaxMoment2Out", "(Tensor) Output max second moment");

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
    The Amsgrad optimzier uses an optimization described at the end
    of section 3 of `Amsgrad paper <https://openreview.net/pdf?id=ryQu7f-RZ>`_ ,
    it can dynamically adjusts the learning rate of each parameter using
    the 1st moment estimates and the 2nd moment estimates of the gradient.

    It is a variant of the Adam optimizer. It considers the long term memory of past gradients by keep tracking the maximum
    of 2nd exponential moving average moment estimates of the gradient.

    Amsgrad update rule:

    $$
    moment\_1\_out & = {\\beta}_1 * moment\_1 + (1 - {\\beta}_1) * grad

    moment\_2\_out & = {\\beta}_2 * moment\_2 + (1 - {\\beta}_2) * grad * grad

    max_moment\_2\_out & = max{max_moment\_2, moment\_2\_out}

    learning\_rate & = learning\_rate * \\
                          \\frac{\sqrt{1 - {\\beta}_2^t}}{1 - {\\beta}_1^t}

    param\_out & = param - learning\_rate * \\frac{moment\_1\_out}{\sqrt{max_moment\_2\_out} + \epsilon}
    $$

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(amsgrad, ops::AmsgradOp, ops::AmsgradOpMaker);
REGISTER_OP_CPU_KERNEL(
    amsgrad, ops::AmsgradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AmsgradOpKernel<paddle::platform::CPUDeviceContext, double>);
