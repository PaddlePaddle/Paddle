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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

class LerpOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class LerpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of lerp op.");
    AddInput("Y", "(Tensor), The input tensor of lerp op.");
    AddInput("Weight", "(Tensor, optional), The input tensor of lerp op.");
    AddOutput("Out", "(Tensor), The output tensor of lerp op.");
    AddComment(R"DOC(
Lerp Operator.

This operator is used to do a linear interpolation of input $X$ and $Y$ with $Weight$.

The equation is:

$$Out = X + Weight * (Y - X)$$

Both the input $X$ and $Y$ can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input $X$.

)DOC");
  }
};

class LerpGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("Y"))) {
      ctx->SetOutputDim(framework::GradVarName("Y"), ctx->GetInputDim("Y"));
    }
  }
};

template <typename T>
class LerpOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("lerp_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Weight", this->Input("Weight"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(LerpInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(lerp, LerpInferShapeFunctor,
                            PD_INFER_META(phi::LerpInferMeta));
REGISTER_OPERATOR(
    lerp, paddle::operators::LerpOp, paddle::operators::LerpOpMaker,
    paddle::operators::LerpOpGradMaker<paddle::framework::OpDesc>,
    paddle::operators::LerpOpGradMaker<paddle::imperative::OpBase>,
    paddle::operators::LerpInplaceInferer, LerpInferShapeFunctor);

REGISTER_OPERATOR(lerp_grad, paddle::operators::LerpGradOp);
