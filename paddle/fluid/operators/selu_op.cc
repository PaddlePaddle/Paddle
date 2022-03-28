/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class SeluOp : public framework::OperatorWithKernel {
 public:
  SeluOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class SeluOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> &GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class SeluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of selu operator.");
    AddOutput("Out", "The output tensor of selu operator.");
    AddAttr<float>("scale",
                   "(float) the default value is 1.0507~. For more "
                   "information about this value, please refer to:"
                   "https://arxiv.org/abs/1706.02515.")
        .SetDefault(1.0507009873554804934193349852946);
    AddAttr<float>("alpha",
                   "(float) the default value is 1.6732~. For more "
                   "information about this value, please refer to:"
                   "https://arxiv.org/abs/1706.02515.")
        .SetDefault(1.6732632423543772848170429916717);
    AddComment(R"DOC(
Selu Operator.

The equation is:
$$
f(x) =\lambda*
\begin{cases}
 \quad \quad   x,  \quad \quad \quad \text{if} \ x > 0 \\
 \alpha * e^x - \alpha,  \qquad  \text{if} \ x <= 0
\end{cases}
$$

The input `X` can carry the LoD (Level of Details) information,
or not. And the output shares the LoD information with input `X`.
)DOC");
  }
};

template <typename T>
class SeluGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("selu_grad");
    grad_op->SetInput("Out", this->Output("Out"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class SeluGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "selu_grad");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "selu_grad");
    auto x_grad_name = framework::GradVarName("X");
    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("Out"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Out"), ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(selu, SeluInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(selu, ops::SeluOp, ops::SeluOpMaker, ops::SeluOpInferVarType,
                  ops::SeluGradMaker<paddle::framework::OpDesc>,
                  ops::SeluGradMaker<paddle::imperative::OpBase>,
                  SeluInferShapeFunctor);

REGISTER_OPERATOR(selu_grad, ops::SeluGradOp);
