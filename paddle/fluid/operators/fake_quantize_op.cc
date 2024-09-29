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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class MovingAverageAbsMaxScaleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "MovingAverageAbsMaxScale");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"),
                   "Output",
                   "OutScale",
                   "MovingAverageAbsMaxScale");

    if (ctx->HasOutput("OutState")) {
      ctx->SetOutputDim("OutState", {1});
    }
    if (ctx->HasOutput("OutAccum")) {
      ctx->SetOutputDim("OutAccum", {1});
    }
    if (ctx->HasOutput("Out")) {
      ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
      ctx->SetOutputDim("OutScale", {1});
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

class MovingAverageAbsMaxScaleOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddInput("InAccum", "Last accum.").AsDispensable();
    AddInput("InState", "Last state.").AsDispensable();
    AddOutput("Out",
              "(Tensor) Output tensor is just equivalent to the input tensor.")
        .AsDispensable();
    AddOutput("OutScale", " Current scale");
    AddOutput("OutState", "(Tensor) state buffer.").AsDispensable();
    AddOutput("OutAccum", "(Tensor) accum buffer.").AsDispensable();
    AddAttr<float>("moving_rate", "(float, default 0.9) moving rate.")
        .SetDefault(0.9);
    AddAttr<bool>("is_test",
                  "(bool, default false) Set true for inference only and false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddComment(R"DOC(
MovingAverageAbsMaxScale operator is only used for calculating the quantization scale.
And it will not quantize the input tensor.

$$scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)$$
$$Out = X$$

)DOC");
  }
};

class StraightThroughEstimatorGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    auto x_grad_name = framework::GradVarName("X");
    OP_INOUT_CHECK(ctx->HasInput(out_grad_name),
                   "Input",
                   out_grad_name,
                   "StraightThroughEstimatorGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(x_grad_name),
                   "Output",
                   x_grad_name,
                   "StraightThroughEstimatorGradOp");

    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim(out_grad_name));
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class StraightThroughEstimatorMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("straight_through_estimator_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    moving_average_abs_max_scale,
    ops::MovingAverageAbsMaxScaleOp,
    ops::MovingAverageAbsMaxScaleOpMaker,
    ops::StraightThroughEstimatorMaker<paddle::framework::OpDesc>,
    ops::StraightThroughEstimatorMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(straight_through_estimator_grad,
                  ops::StraightThroughEstimatorGradOp);

REGISTER_OP_VERSION(fake_channel_wise_quantize_abs_max)
    .AddCheckpoint(
        R"ROC(add new attributes [quant_axis] for applying per-channel "
        "quantization to conv2d_transpose and mul ops.)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "quant_axis", "The axis for quantization.", 0));
REGISTER_OP_VERSION(moving_average_abs_max_scale)
    .AddCheckpoint(
        R"ROC(Incompatible upgrade of output [Out])ROC",
        paddle::framework::compatible::OpVersionDesc().DeleteOutput(
            "Out",
            "Delete output in order to make the inference model not "
            "save moving_average_abs_max_scale operator. This will "
            "make the quantitative model be correctly applied in inference."))
    .AddCheckpoint(R"ROC(Incompatible upgrade of output [Out])ROC",
                   paddle::framework::compatible::OpVersionDesc().NewOutput(
                       "Out",
                       "In order to support dygraph qat, add output again."));
