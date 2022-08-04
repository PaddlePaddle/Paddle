/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

class FillAnyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensor.");
    AddOutput("Out", "Tensor, the tensor filled with input value ");
    AddAttr<float>("value_float", "The float var to fill in Tensor")
        .SetDefault(0);
    AddAttr<int>("value_int", "The int var to fill in Tensor").SetDefault(0);
    AddComment(R"DOC(Fill operator with backward;
                Fill an tensor with `value`. 
                )DOC");
  };
};

class FillAnyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "FillAny");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "FillAny");
    auto x_dims = context->GetInputDim("X");
    context->SetOutputDim("Out", x_dims);
  }
};

class FillAnyGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "mul");
    auto x_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }
};

template <typename T>
class FillAnyGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType(this->ForwardOpType() + "_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(FillAnyOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(FillAnyGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(fill_any,
                  ops::FillAnyOp,
                  ops::FillAnyOpMaker,
                  ops::FillAnyGradOpMaker<paddle::framework::OpDesc>,
                  ops::FillAnyGradOpMaker<paddle::imperative::OpBase>,
                  ops::FillAnyOpInplaceInferer);

REGISTER_OPERATOR(fill_any_grad,
                  ops::FillAnyGradOp,
                  ops::FillAnyGradInplaceInferer);
