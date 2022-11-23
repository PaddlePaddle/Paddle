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

#include <memory>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class ArgsortOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class ArgsortGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*-->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class ArgsortOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of Argsort op.");
    AddOutput("Out",
              "(Tensor) The sorted tensor of Argsort op, with the same "
              "shape as Input(X).");
    AddOutput("Indices",
              "(Tensor) The indices of a tensor giving the sorted order, with "
              "the same shape as Input(X).");
    AddComment(R"DOC(
Argsort operator

Performs sorting on the input tensor along the given axis and outputs two
tensors, Output(Out) and Output(Indices). They reserve the same shape
with Input(X), and Output(Out) represents the sorted tensor while
Output(Indices) gives the sorted order along the given axis Attr(axis).

 )DOC");
    AddAttr<int>("axis",
                 "(int, default -1) The axis along which to sort the tensor. "
                 "When axis < 0, the actual axis will be the |axis|'th "
                 "counting backwards. Default -1, the last dimension.")
        .SetDefault(-1);
    AddAttr<bool>(
        "descending",
        "(bool, default false) The descending attribute is a flag to tell"
        "algorithm how to sort the input data."
        "If descending is true, will sort by descending order,"
        "else if false, sort by ascending order. Default value is false.")
        .SetDefault(false);
  }
};

template <typename T>
class ArgsortGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("argsort_grad");
    op->SetInput("Indices", this->Output("Indices"));
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ArgsortGradNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(argsort,
                            ArgsortInferShapeFunctor,
                            PD_INFER_META(phi::ArgsortInferMeta));
REGISTER_OPERATOR(argsort,
                  ops::ArgsortOp,
                  ops::ArgsortOpMaker,
                  ops::ArgsortGradOpMaker<paddle::framework::OpDesc>,
                  ops::ArgsortGradOpMaker<paddle::imperative::OpBase>,
                  ArgsortInferShapeFunctor);
REGISTER_OPERATOR(argsort_grad,
                  ops::ArgsortGradOp,
                  ops::ArgsortGradNoNeedBufferVarsInferer);
