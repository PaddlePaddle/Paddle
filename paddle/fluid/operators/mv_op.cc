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

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class MVOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The matrix input of mv op");
    AddInput("Vec", "The vector input of mv op");
    AddOutput("Out", "The output of mv op");
    AddComment(R"DOC(
MV Operator.

This operator is used to perform matrix vector multiplication
of the input tensors `X` and `Vec`.
)DOC");
  }
};

class MVOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
};

template <typename T>
class MVOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("mv_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Vec", this->Input("Vec"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Vec"), this->InputGrad("Vec"));
  }
};

class MVOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "mv");
    OP_INOUT_CHECK(context->HasInput("Vec"), "Input", "Vec", "mv");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "mv");
    auto x_dims = context->GetInputDim("X");
    auto vec_dims = context->GetInputDim("Vec");

    auto x_grad_name = framework::GradVarName("X");
    auto vec_grad_name = framework::GradVarName("Vec");

    if (context->HasOutput(x_grad_name)) {
      context->SetOutputDim(x_grad_name, x_dims);
    }
    if (context->HasOutput(vec_grad_name)) {
      context->SetOutputDim(vec_grad_name, vec_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(mv, MvInferShapeFunctor,
                            PD_INFER_META(phi::MvInferMeta));

REGISTER_OPERATOR(mv, ops::MVOp, ops::MVOpMaker,
                  ops::MVOpGradMaker<paddle::framework::OpDesc>,
                  ops::MVOpGradMaker<paddle::imperative::OpBase>,
                  MvInferShapeFunctor);
REGISTER_OPERATOR(mv_grad, ops::MVOpGrad);
