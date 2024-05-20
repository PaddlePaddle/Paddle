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

#include "paddle/fluid/operators/fake_quantize_op.h"

#include <algorithm>
#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/impl/clip_kernel_impl.h"

namespace paddle {
namespace operators {

template <typename T>
struct Compare {
 public:
  bool operator()(const T a, const T b) { return (std::abs(a) < std::abs(b)); }
};

template <typename T>
struct ClipAndFakeQuantDequantFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext &ctx,
                  const phi::DenseTensor &in,
                  const phi::DenseTensor &scale,
                  const int bin_cnt,
                  const int round_type,
                  phi::DenseTensor *out) {
    T s = scale.data<T>()[0];
    T inv_s = phi::funcs::inverse(s);

    phi::Transform<phi::CPUContext> trans;
    if (round_type == 0) {
      trans(ctx,
            in.data<T>(),
            in.data<T>() + in.numel(),
            out->mutable_data<T>(ctx.GetPlace()),
            phi::funcs::QuantTensorFunctor<T>(static_cast<T>(bin_cnt), inv_s));
      auto out_e = phi::EigenVector<T>::Flatten(*out);
      out_e.device(*ctx.eigen_device()) = out_e * s / static_cast<T>(bin_cnt);
    } else {
      trans(ctx,
            in.data<T>(),
            in.data<T>() + in.numel(),
            out->mutable_data<T>(ctx.GetPlace()),
            phi::ClipFunctor<T>(-s, s));
      auto out_e = phi::EigenVector<T>::Flatten(*out);
      out_e.device(*ctx.eigen_device()) =
          (bin_cnt * inv_s * out_e).round() * s / static_cast<T>(bin_cnt);
    }
  }
};
template struct ClipAndFakeQuantDequantFunctor<phi::CPUContext, float>;

class FakeQuantOrWithDequantAbsMaxOp : public framework::OperatorWithKernel {
 public:
  FakeQuantOrWithDequantAbsMaxOp(const std::string &type,
                                 const framework::VariableNameMap &inputs,
                                 const framework::VariableNameMap &outputs,
                                 const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "FakeQuantOrWithDequantAbsMaxOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"),
                   "Output",
                   "Out",
                   "FakeQuantOrWithDequantAbsMaxOp");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"),
                   "Output",
                   "OutScale",
                   "FakeQuantOrWithDequantAbsMaxOp");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.device_context().GetPlace());
  }
};

class FakeQuantOrWithDequantAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddOutput("Out",
              "(Tensor) Output of quantized low level tensor, "
              "but also saved as float data type.");
    AddOutput("OutScale", "(Tensor) Current scale");
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int &bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                            true,
                            phi::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddComment(R"DOC(
This is a Base Op which supports FakeQuantAbsMaxOpMaker and FakeQuantDequantAbsMaxOpMaker.
FakeQuantAbsMaxOp operator is used in the dynamic quantization.

$$scale = max(abs(X))$$
$$range = 2^{bit_length - 1} - 1$$
$$Out = round(X/scale * range)$$

FakeQuantDequantAbsMaxOp operator does the abs_max quantization and then dequantization.

$$scale = max(abs(X))$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out = round(X/scale * range) * scale / range$$

)DOC");
  }
};

class FakeQuantOrWithDequantMovingAverageAbsMaxOp
    : public framework::OperatorWithKernel {
 public:
  FakeQuantOrWithDequantMovingAverageAbsMaxOp(
      const std::string &type,
      const framework::VariableNameMap &inputs,
      const framework::VariableNameMap &outputs,
      const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"),
                   "Input",
                   "X",
                   "FakeQuantOrWithDequantMovingAverageAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("Out"),
                   "Output",
                   "Out",
                   "FakeQuantOrWithDequantMovingAverageAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"),
                   "Output",
                   "OutScale",
                   "FakeQuantOrWithDequantMovingAverageAbsMax");
    if (ctx->HasOutput("OutState")) {
      ctx->SetOutputDim("OutState", {1});
    }
    if (ctx->HasOutput("OutAccum")) {
      ctx->SetOutputDim("OutAccum", {1});
    }
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.device_context().GetPlace());
  }
};

class FakeQuantOrWithDequantMovingAverageAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddInput("InScale", "Last scale.");
    AddInput("InAccum", "Last accum.").AsDispensable();
    AddInput("InState", "Last state.").AsDispensable();
    AddOutput("Out", "(Tensor) Output of quantized low level tensor.");
    AddOutput("OutScale", " Current scale");
    AddOutput("OutState", "(Tensor) state buffer.").AsDispensable();
    AddOutput("OutAccum", "(Tensor) accum buffer.").AsDispensable();
    AddAttr<float>("moving_rate", "(float, default 0.9) moving rate.")
        .SetDefault(0.9);
    AddAttr<int>("bit_length", "(int, default 8), quantization bit number.")
        .SetDefault(8)
        .AddCustomChecker([](const int &bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16,
                            true,
                            phi::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddComment(R"DOC(
This is a Base Op which supports FakeQuantMovingAverageAbsMaxOp and FakeQuantDequantMovingAverageAbsMaxOp.
FakeQuantMovingAverageAbsMaxOp operator is used in the static quantization.

$$scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out = round(X/scale * range)$$

FakeQuantDequantMovingAverageAbsMaxOp operator does the moving_average_abs_max quant and then dequant.

$$scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out = round(X/scale * range) * scale / range$$

)DOC");
  }
};

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
using CPU = phi::CPUContext;

REGISTER_OPERATOR(
    fake_quantize_dequantize_abs_max,
    ops::FakeQuantOrWithDequantAbsMaxOp,
    ops::FakeQuantOrWithDequantAbsMaxOpMaker,
    ops::StraightThroughEstimatorMaker<paddle::framework::OpDesc>,
    ops::StraightThroughEstimatorMaker<paddle::imperative::OpBase>);
PD_REGISTER_STRUCT_KERNEL(fake_quantize_dequantize_abs_max,
                          CPU,
                          ALL_LAYOUT,
                          ops::FakeQuantizeDequantizeAbsMaxKernel,
                          float) {}

REGISTER_OPERATOR(
    fake_quantize_dequantize_moving_average_abs_max,
    ops::FakeQuantOrWithDequantMovingAverageAbsMaxOp,
    ops::FakeQuantOrWithDequantMovingAverageAbsMaxOpMaker,
    ops::StraightThroughEstimatorMaker<paddle::framework::OpDesc>,
    ops::StraightThroughEstimatorMaker<paddle::imperative::OpBase>);
PD_REGISTER_STRUCT_KERNEL(fake_quantize_dequantize_moving_average_abs_max,
                          CPU,
                          ALL_LAYOUT,
                          ops::FakeQuantizeDequantizeMovingAverageAbsMaxKernel,
                          float) {}

REGISTER_OPERATOR(
    moving_average_abs_max_scale,
    ops::MovingAverageAbsMaxScaleOp,
    ops::MovingAverageAbsMaxScaleOpMaker,
    ops::StraightThroughEstimatorMaker<paddle::framework::OpDesc>,
    ops::StraightThroughEstimatorMaker<paddle::imperative::OpBase>);
PD_REGISTER_STRUCT_KERNEL(moving_average_abs_max_scale,
                          CPU,
                          ALL_LAYOUT,
                          ops::MovingAverageAbsMaxScaleKernel,
                          float) {}

REGISTER_OPERATOR(straight_through_estimator_grad,
                  ops::StraightThroughEstimatorGradOp);
PD_REGISTER_STRUCT_KERNEL(straight_through_estimator_grad,
                          CPU,
                          ALL_LAYOUT,
                          ops::StraightThroughEstimatorGradKernel,
                          float) {}

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
