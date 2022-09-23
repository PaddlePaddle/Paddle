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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

class FillAnyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensor.");
    AddOutput("Out", "Tensor, the tensor filled with input value ");
    AddAttr<paddle::experimental::Scalar>(
        "value", "generic value to fill into the tensoer.")
        .SetDefault(0.0);
    AddComment(R"DOC(Fill operator with backward;
                Fill an tensor with `value`.
                )DOC");
  };
};

class FillAnyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class FillAnyGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
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

DECLARE_INFER_SHAPE_FUNCTOR(fill_any,
                            FillAnyInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(fill_any_grad,
                            FillAnyGradInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(fill_any,
                  ops::FillAnyOp,
                  ops::FillAnyOpMaker,
                  ops::FillAnyGradOpMaker<paddle::framework::OpDesc>,
                  ops::FillAnyGradOpMaker<paddle::imperative::OpBase>,
                  ops::FillAnyOpInplaceInferer,
                  FillAnyInferShapeFunctor);

REGISTER_OPERATOR(fill_any_grad,
                  ops::FillAnyGradOp,
                  ops::FillAnyGradInplaceInferer,
                  FillAnyGradInferShapeFunctor);

REGISTER_OP_VERSION(fill_any).AddCheckpoint(
    R"ROC(
Upgrade fill_any, change the type of attribute value to Scalar to support
generic type.
              )ROC",
    paddle::framework::compatible::OpVersionDesc()
        .DeleteAttr("value_float",
                    "remove plain attribute in favor of generics")
        .DeleteAttr("value_int", "remove plain attribute in favor of generics")
        .NewAttr("value", "generic value", 0.0));
