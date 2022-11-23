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
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class FillIDiagonalOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(Fill replace operator
                Fill the diagonal of an tensor with 'value'.
                )DOC");
    AddInput("X", "(Tensor) The input tensor.");
    AddOutput("Out",
              "Tensor, the output tensor, with the same shape and data type "
              "as input(x)");
    AddAttr<float>(
        "value",
        "The float values of tensor, whose dim is one, and no need of grad")
        .SetDefault(0);
    AddAttr<bool>("wrap",
                  "the diagonal 'wrapped' after N columns for tall matrices")
        .SetDefault(false);
    AddAttr<int>("offset",
                 "offset of diagonal, zero means no offset, positive means "
                 "offset to up-right corner; negtive means offset to "
                 "bottom-left corner")
        .SetDefault(0);
  }
};

class FillIDiagonalOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FillIDiagonalOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_type = ctx->GetInputType("X", 0);
    auto data_type = ctx->GetInputDataType("X", 0);
    ctx->SetOutputType("Out", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType("Out", data_type, framework::ALL_ELEMENTS);
  }
};

class FillIDiagonalGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    // Note: don't get data type from ctx.Input<phi::DenseTensor>("Input");
    auto dtype = framework::TransToProtoVarType(
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"))->type());
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

template <typename T>
class FillIDiagonalGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("fill_diagonal_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(FillIDiagonalOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(FillIDiagonalGradOpInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(fill_diagonal,
                            FillDiagonalShapeFunctor,
                            PD_INFER_META(phi::FillDiagonalInferMeta));

DECLARE_INFER_SHAPE_FUNCTOR(fill_diagonal_grad,
                            FillDiagonalGradShapeFunctor,
                            PD_INFER_META(phi::FillDiagonalGradInferMeta));

REGISTER_OPERATOR(fill_diagonal,
                  ops::FillIDiagonalOp,
                  ops::FillIDiagonalGradOpMaker<paddle::framework::OpDesc>,
                  ops::FillIDiagonalGradOpMaker<paddle::imperative::OpBase>,
                  ops::FillIDiagonalOpMaker,
                  ops::FillIDiagonalOpInplaceInferer,
                  ops::FillIDiagonalOpVarTypeInference,
                  FillDiagonalShapeFunctor);

REGISTER_OPERATOR(fill_diagonal_grad,
                  ops::FillIDiagonalGradOp,
                  ops::FillIDiagonalGradOpInplaceInferer,
                  FillDiagonalGradShapeFunctor);
