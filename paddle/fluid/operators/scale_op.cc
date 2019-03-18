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

#include <memory>
#include <string>

#include "paddle/fluid/operators/detail/safe_ref.h"

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

Apply scaling and bias addition to the input tensor.

if bias_after_scale=True:

$$Out = scale*X + bias$$

else:

$$Out = scale*(X + bias)$$
)DOC");
    AddAttr<float>("scale", "The scaling factor of the scale operator.")
        .SetDefault(1.0);
    AddAttr<float>("bias", "The bias of the scale operator.").SetDefault(0.0);
    AddAttr<bool>(
        "bias_after_scale",
        "Apply bias addition after or before scaling. It is useful for "
        "numeric stability in some circumstances.")
        .SetDefault(true);
  }
};

class ScaleOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto &in_var_name = ctx->Input("X").front();
    auto out_var_name = ctx->Output("Out").front();

    if (in_var_name != out_var_name) {
      ctx->SetType(out_var_name, ctx->GetType(in_var_name));
      ctx->SetDataType(out_var_name, ctx->GetDataType(in_var_name));
    }
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
    grad_op->SetAttr("bias", 0.0f);
    grad_op->SetAttr("bias_after_scale", true);
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

using ScaleOpInplace = framework::SingleOpInplaceInToOut;
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(scale, ops::ScaleOp, ops::ScaleOpMaker, ops::ScaleGradMaker,
                  ops::ScaleOpVarTypeInference, ops::ScaleOpInplace);
REGISTER_OP_CPU_KERNEL(
    scale, ops::ScaleKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int64_t>);
