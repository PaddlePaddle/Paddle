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

#include "paddle/fluid/operators/scale_op.h"
#include <string>

namespace paddle {
namespace operators {

class ScaleOp : public framework::OperatorWithKernel {
 public:
  ScaleOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ScaleOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ScaleOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ScaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of scale operator.");
    AddOutput("Out", "(Tensor) Output tensor of scale operator.");
    AddComment(R"DOC(
**Scale operator**

Multiply the input tensor with a float scalar to scale the input tensor.

$$Out = scale*X$$
)DOC");
    AddAttr<float>("scale", "The scaling factor of the scale operator.")
        .SetDefault(1.0);
  }
};

class ScaleGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("scale");
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttr("scale", GetAttr("scale"));
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(scale, ops::ScaleOp, ops::ScaleOpMaker, ops::ScaleGradMaker);
REGISTER_OP_CPU_KERNEL(
    scale, ops::ScaleKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int64_t>);
