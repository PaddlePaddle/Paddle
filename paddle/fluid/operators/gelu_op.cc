/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class GeluOp : public framework::OperatorWithKernel {
 public:
  GeluOp(const std::string &type,
         const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, data_type)) {
      return framework::OpKernelType(data_type,
                                     ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(data_type, ctx.GetPlace());
  }
};

class GeluGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")),
        true,
        platform::errors::InvalidArgument(
            "Input(%s) of GeluGradOp should not be null.", "DOut"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                      true,
                      platform::errors::InvalidArgument(
                          "Input(%s) of GeluGradOp should not be null.", "X"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")),
        true,
        platform::errors::InvalidArgument(
            "Output(%s) of GeluGradOp should not be null.", "DX"));
    auto x_grad_name = framework::GradVarName("X");
    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ x_grad_name);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, data_type)) {
      return framework::OpKernelType(data_type,
                                     ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(data_type, ctx.GetPlace());
  }
};

class GeluOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of Gelu operator");
    AddOutput("Out", "Output of Gelu operator");
    AddAttr<bool>("approximate",
                  "(bool, default false) use approximation of gelu")
        .SetDefault(false);
    AddComment(R"DOC(
Gelu Activation Operator.

For more details, please refer to [Gaussian Error Linear Units](https://arxiv.org/pdf/1606.08415.pdf).

when using approximation
$out = \\frac{1}{2}x(1+tanh(\\sqrt{\\frac{2}{\\pi}}(x+0.044715x^{3}))$

or else
$out = \\frac{1 + erf(\\frac{x}{\\sqrt{2}})}{2} x$

)DOC");
  }
};

template <typename T>
class GeluGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("gelu_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(gelu,
                            GeluInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));
REGISTER_OPERATOR(gelu,
                  ops::GeluOp,
                  ops::GeluOpMaker,
                  ops::GeluGradOpMaker<paddle::framework::OpDesc>,
                  ops::GeluGradOpMaker<paddle::imperative::OpBase>,
                  GeluInferShapeFunctor);
REGISTER_OPERATOR(gelu_grad, ops::GeluGradOp);
