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
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

class TruncOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class TruncOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of trunc op.");
    AddOutput("Out", "(Tensor), The output tensor of trunc op.");
    AddComment(R"DOC(
Trunc Operator.
Returns a new tensor with the truncated integer values  of input.
$$out = trunc(x)$$
)DOC");
  }
};

class TruncGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class TruncGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("trunc_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DELCARE_INFER_SHAPE_FUNCTOR(trunc, TruncInferShapeFunctor,
                            PT_INFER_META(pten::TruncInferMeta));

REGISTER_OPERATOR(trunc, ops::TruncOp, ops::TruncOpMaker,
                  ops::TruncGradOpMaker<paddle::framework::OpDesc>,
                  ops::TruncGradOpMaker<paddle::imperative::OpBase>,
                  TruncInferShapeFunctor);

DELCARE_INFER_SHAPE_FUNCTOR(trunc_grad, TruncGradInferShapeFunctor,
                            PT_INFER_META(pten::TruncGradInferMeta));
REGISTER_OPERATOR(trunc_grad, ops::TruncGradOp, TruncGradInferShapeFunctor);
