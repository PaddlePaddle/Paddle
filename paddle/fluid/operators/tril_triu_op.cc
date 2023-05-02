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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class TrilTriuOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class TrilTriuOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Tensor, the input of tril_triu op");
    AddOutput("Out",
              "Tensor, the output tensor, with the same shape and data type as "
              "input(x)");
    AddAttr<int>("diagonal", "int number, the diagonal to consider.")
        .SetDefault(0);
    AddAttr<bool>("lower", "boolnumber, lower triangular or upper triangular.");
    AddComment(R"DOC(
TrilTriu Operator.

The tril operator returns the lower triangular part of the matrix (2-D tensor)
or batch of matrices $input$. The lower triangular part of the matrix is defined
as the elements on and below the diagonal.
The triu operator returns the upper triangular part of a matrix (2-D tensor)
or batch of matrices $input$. The upper triangular part of the matrix is defined
as the elements on and above the diagonal.
The other elements of the result tensor out are set to 0.

The argument diagonal controls which diagonal to consider, default value is 0.

)DOC");
  }
};

class TrilTriuGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")),
                      true,
                      platform::errors::NotFound(
                          "Input(Out@GRAD) of TrilTriuOp should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")),
                      true,
                      platform::errors::NotFound(
                          "Output(X@Grad) of TrilTriuOp should not be null"));
    ctx->SetOutputDim(framework::GradVarName("X"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
  }
};

template <typename T>
class TrilTriuGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("tril_triu_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
DECLARE_INFER_SHAPE_FUNCTOR(tril_triu,
                            TrilTriuInferShapeFunctor,
                            PD_INFER_META(phi::TrilTriuInferMeta));
REGISTER_OPERATOR(tril_triu,
                  ops::TrilTriuOp,
                  ops::TrilTriuOpMaker,
                  ops::TrilTriuGradOpMaker<paddle::framework::OpDesc>,
                  ops::TrilTriuGradOpMaker<paddle::imperative::OpBase>,
                  TrilTriuInferShapeFunctor);
REGISTER_OPERATOR(tril_triu_grad, ops::TrilTriuGradOp);
