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
#include <string>

namespace paddle {
namespace operators {

class FakeQuantizeOp : public framework::OperatorWithKernel {
 public:
  FakeQuantizeOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of FakeQuantizeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FakeQuantizeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("OutMovingScale"), "");
    // if (ctx->HasInput("InMovingScale")) {
    ctx->SetOutputDim("OutMovingScale", ctx->GetInputDim("InMovingScale"));
    //}
    // if (ctx->HasInput("InScales")) {
    PADDLE_ENFORCE(ctx->HasOutput("OutScales"), "");
    ctx->SetOutputDim("OutScales", ctx->GetInputDim("InScales"));
    // PADDLE_ENFORCE_EQ(ctx->Inputs("InScales")[0],
    // ctx->Outputs("OutScales")[0],
    //                  "Mean and MeanOut should share the same memory");
    //}
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class FakeQuantizeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of scale operator.");
    AddInput("InScales", "(Tensor) scale buffer").AsDispensable();
    AddInput("InMovingScale", "last scale").AsDispensable();
    AddInput("InCurrentIter", "last Iter").AsDispensable();
    AddOutput("Out", "(Tensor) Output tensor of scale operator.");
    AddOutput("OutScales", "(Tensor) scale buffer").AsDispensable();
    AddOutput("OutMovingScale", " current scale");
    AddOutput("OutCurrentIter", "current Iter").AsDispensable();
    AddComment(R"DOC(
FakeQuantize operator

$$Out = scale*X$$
)DOC");
    AddAttr<std::string>("quantize_type",
                         "(string, default abs_max)"
                         "The scaling tpe of the quantize operator.")
        .SetDefault("abs_max");
    AddAttr<int>("window_size", "(int, default 10000)").SetDefault(10000);
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int &bit_length) {
          PADDLE_ENFORCE(bit_length >= 1 && bit_length <= 16,
                         "'bit_length' should be between 1 and 16.");
        });
    AddAttr<bool>("is_test", "").SetDefault(false);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(fake_quantize, ops::FakeQuantizeOp, ops::FakeQuantizeOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    fake_quantize,
    ops::FakeQuantizeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FakeQuantizeKernel<paddle::platform::CPUDeviceContext, double>);
