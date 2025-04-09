/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle::operators {
class QuantizeLinearOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "QuantizeLinear");
    OP_INOUT_CHECK(ctx->HasInput("Scale"), "Input", "Scale", "QuantizeLinear");
    OP_INOUT_CHECK(
        ctx->HasInput("ZeroPoint"), "Input", "ZeroPoint", "QuantizeLinear");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "QuantizeLinear");
    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    int quant_axis = ctx->Attrs().Get<int>("quant_axis");
    if (ctx->HasOutput("OutScale")) {
      if (quant_axis < 0) {
        ctx->SetOutputDim("OutScale", {1});
      } else {
        ctx->SetOutputDim("OutScale", {ctx->GetInputDim("X")[quant_axis]});
      }
    }
    if (ctx->HasOutput("OutState")) {
      ctx->SetOutputDim("OutState", {1});
    }
    if (ctx->HasOutput("OutAccum")) {
      ctx->SetOutputDim("OutAccum", {1});
    }
    ctx->ShareLoD("X", /*->*/ "Y");
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};
class QuantizeLinearOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddInput("Scale", "(Tensor) Input is float data type.");
    AddInput("ZeroPoint", "(Tensor) Input is float data type.");
    AddOutput("Y",
              "(Tensor) Output of quantized low level tensor, "
              "but also saved as float data type.");
    AddInput("InAccum", "Last accum.")
        .AsDispensable()
        .AsExtra();  // only qat use
    AddInput("InState", "Last state.")
        .AsDispensable()
        .AsExtra();  // only qat use
    AddOutput("OutState", "(Tensor) state buffer.")
        .AsDispensable()
        .AsExtra();  // only qat use
    AddOutput("OutAccum", "(Tensor) accum buffer.")
        .AsDispensable()
        .AsExtra();  // only qat use
    AddOutput("OutScale", "(Tensor) Current scale")
        .AsDispensable()
        .AsExtra();  // only qat use
    AddAttr<int>("quant_axis",
                 "(int, default 0) The axis for quantization. "
                 "For conv2d, depthwise_conv2d, conv2d_transpose "
                 "and mul, the quant_axis is equal to the cout axis.")
        .SetDefault(0);
    AddAttr<int>("bit_length", "(int, default 8)").SetDefault(8);
    AddAttr<int>("qmin", "(int, default -128)").SetDefault(-128);
    AddAttr<int>("qmax", "(int, default 127)").SetDefault(127);
    AddAttr<int>(
        "round_type",
        "(int, default 0) The round type of fp32 to int."
        "0: rounding to nearest ties to even. Eg: round(1.5)=2, "
        "round(2.5)=2"
        "1: rounding to nearest ties away from zero. Eg: round(1.5)=2, "
        "round(2.5)=3")
        .SetDefault(0);
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(true);
    AddAttr<bool>(
        "only_observer",
        "(bool, default false) Whether to only observer or not. If "
        "only_observer=false, it will calculate fake quant or dequant output. "
        "If only_observer=true, it will only calibrate scale information.")
        .SetDefault(false);
    AddComment(R"DOC(
The scale of QuantizeLinear operator is a vector.
In detail, each channel of the input X has a scale value.
$$scale_c = max(abs(X_c))$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out_c = round(\frac{X_c * range} {scale_c})$$
In above three formulas, the range value of c is as follow:
$$0 \leq c \lt \ the\ channel\ number\ of\ X$$
)DOC");
  }
};
}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    quantize_linear,
    ops::QuantizeLinearOp,
    ops::QuantizeLinearOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    dequantize_linear,
    ops::QuantizeLinearOp,
    ops::QuantizeLinearOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
