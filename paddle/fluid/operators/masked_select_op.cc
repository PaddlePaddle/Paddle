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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class MaskedSelectOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class MaskedSelectOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("Mask",
             "The mask of Input Tensor to be selected which is a bool Tensor.");
    AddOutput(
        "Y",
        "The returned tensor, the data type "
        "is same as input, will be on the same device with the input Tensor.");
    AddComment(R"DOC(
Size Operator.

Return a new 0-D tensor which indexes the indexed tensor according
the mask which is a tensor withe data type bool.
)DOC");
  }
};

class MaskedSelectOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Input",
                   "Input", "MaskedSelect");
    OP_INOUT_CHECK(ctx->HasInput("Mask"), "Input", "Mask", "MaskedSelect");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*-->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Y")),
                                   ctx.device_context());
  }
};

template <typename T>
class MaskedSelectGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("masked_select_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Mask", this->Input("Mask"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(MaskedSelectedGradNoNeedBufferVarsInferer,
                                    "X");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(masked_select, MaksedSelectInferShapeFunctor,
                            PD_INFER_META(phi::MaskedSelectInferMeta));

REGISTER_OPERATOR(masked_select, ops::MaskedSelectOp, ops::MaskedSelectOpMaker,
                  ops::MaskedSelectGradOpMaker<paddle::framework::OpDesc>,
                  ops::MaskedSelectGradOpMaker<paddle::imperative::OpBase>,
                  MaksedSelectInferShapeFunctor);
REGISTER_OPERATOR(masked_select_grad, ops::MaskedSelectOpGrad,
                  ops::MaskedSelectedGradNoNeedBufferVarsInferer);
