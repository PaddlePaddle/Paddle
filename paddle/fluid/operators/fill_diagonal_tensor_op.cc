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
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class FillDiagonalTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(Fill replace operator
                Fill the diagonal of an tensor with `Y` Tensor.
                )DOC");
    AddInput("X", "(Tensor) The input tensor.");
    AddInput("Y", "(Tensor) The input tensor to fill in.");
    AddOutput("Out",
              "Tensor, the output tensor, with the same shape and data type "
              "as input(x)");
    AddAttr<int>("dim1", "the first dim to figure out the diagonal")
        .SetDefault(0);
    AddAttr<int>("dim2", "the second dim to figure out the diagonal")
        .SetDefault(1);
    AddAttr<int64_t>("offset",
                     "offset of diagonal, zero means no offset, positive means "
                     "offset to up-right corner; negtive means offset to "
                     "bottom-left corner")
        .SetDefault(0);
  }
};

class FillDiagonalTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FillDiagonalTensorOpVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_type = ctx->GetInputType("X", 0);
    auto data_type = ctx->GetInputDataType("X", 0);
    ctx->SetOutputType("Out", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType("Out", data_type, framework::ALL_ELEMENTS);
  }
};

class FillDiagonalTensorGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    // Note: don't get data type from ctx.Input<phi::DenseTensor>("Input");
    auto dtype =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"))->type();
    return framework::OpKernelType(framework::TransToProtoVarType(dtype),
                                   ctx.GetPlace());
  }
};

template <typename T>
class FillDiagonalTensorGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("fill_diagonal_tensor_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(FillDiagonalTensorOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(FillDiagonalTensorGradOpInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(fill_diagonal_tensor,
                            FillDiagonalTensorInferShapeFunctor,
                            PD_INFER_META(phi::FillDiagonalTensorInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(
    fill_diagonal_tensor_grad,
    FillDiagonalTensorGradInferShapeFunctor,
    PD_INFER_META(phi::FillDiagonalTensorGradInferMeta));

REGISTER_OPERATOR(
    fill_diagonal_tensor,
    ops::FillDiagonalTensorOp,
    ops::FillDiagonalTensorGradOpMaker<paddle::framework::OpDesc>,
    ops::FillDiagonalTensorGradOpMaker<paddle::imperative::OpBase>,
    ops::FillDiagonalTensorOpMaker,
    ops::FillDiagonalTensorOpInplaceInferer,
    ops::FillDiagonalTensorOpVarTypeInference,
    FillDiagonalTensorInferShapeFunctor);

REGISTER_OPERATOR(fill_diagonal_tensor_grad,
                  ops::FillDiagonalTensorGradOp,
                  ops::FillDiagonalTensorGradOpInplaceInferer,
                  FillDiagonalTensorGradInferShapeFunctor);
