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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class SquaredL2NormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class SquaredL2NormGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("squared_l2_norm_grad");

    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("X", this->Input("X"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));

    op->SetAttrMap(this->Attrs());
  }
};

class SquaredL2NormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class SquaredL2NormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of squared_l2_norm op.");
    AddOutput("Out", "(Scalar) The output of squared_l2_norm op.");
    AddComment(R"DOC(
SquaredL2Norm Operator.

Computes the squared L2 norm of a tensor.

$$Out = \sum_{i} X_{i}^2$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(squared_l2_norm,
                            SquaredL2NormInferShapeFunctor,
                            PD_INFER_META(phi::SquaredL2NormInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(squared_l2_norm_grad,
                            SquaredL2NormGradInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(squared_l2_norm,
                  ops::SquaredL2NormOp,
                  ops::SquaredL2NormOpMaker,
                  ops::SquaredL2NormGradOpMaker<paddle::framework::OpDesc>,
                  ops::SquaredL2NormGradOpMaker<paddle::imperative::OpBase>,
                  SquaredL2NormInferShapeFunctor);

REGISTER_OPERATOR(squared_l2_norm_grad,
                  ops::SquaredL2NormGradOp,
                  SquaredL2NormGradInferShapeFunctor);
