// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class RenormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  using DDim = paddle::framework::DDim;
};

class RenormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of renorm op.");
    AddOutput("Out", "(Tensor), The output tensor of renorm op.");
    AddAttr<float>("p", "(float, norm's power");
    AddAttr<int>("axis",
                 "int,the dimension to slice over to get the sub-tensors");
    AddAttr<float>("max_norm", "(float, the norm upper-bound");
    AddComment(R"DOC(
Renorm Operator.

This operator is used to scale tensor sliced by axis if its p-norm execeeds maxnorm

)DOC");
  }
};

class RenormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

template <typename T>
class RenormGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("renorm_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(renorm,
                            RenormInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(renorm_grad,
                            RenormGradInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(renorm,
                  ops::RenormOp,
                  ops::RenormOpMaker,
                  ops::RenormGradMaker<paddle::framework::OpDesc>,
                  ops::RenormGradMaker<paddle::imperative::OpBase>,
                  RenormInferShapeFunctor)

REGISTER_OPERATOR(renorm_grad, ops::RenormGradOp, RenormGradInferShapeFunctor);
