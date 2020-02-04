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

#include "paddle/fluid/operators/argsort_op.h"
#include <memory>

namespace paddle {
namespace operators {

class ArgsortOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ArgsortOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ArgsortOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Indices"),
                   "Output(Indices) of ArgsortOp should not be null.");

    auto in_dims = ctx->GetInputDim("X");
    int axis = ctx->Attrs().Get<int>("axis");

    auto num_dims = in_dims.size();
    PADDLE_ENFORCE(axis < num_dims,
                   "Attr(axis) %d of ArgsortOp is out of bounds for Input(X)'s "
                   "rank %d.",
                   axis, num_dims);
    PADDLE_ENFORCE(axis >= -num_dims,
                   "Attr(axis) %d of ArgsortOp must be not less than "
                   "-rank(Input(X)) (%d).",
                   axis, num_dims);

    ctx->ShareDim("X", "Out");
    ctx->ShareDim("X", "Indices");
    ctx->ShareLoD("X", "Out");
    ctx->ShareLoD("X", "Indices");
  }
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
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("argsort_grad");
    op->SetInput("Indices", this->Output("Indices"));
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(ArgsortGradNoNeedBufferVarInference, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(argsort, ops::ArgsortOp, ops::ArgsortOpMaker,
                  ops::ArgsortGradOpMaker<paddle::framework::OpDesc>,
                  ops::ArgsortGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(argsort_grad, ops::ArgsortGradOp,
                  ops::ArgsortGradNoNeedBufferVarInference);
REGISTER_OP_CPU_KERNEL(argsort,
                       ops::ArgsortKernel<paddle::platform::CPUPlace, float>,
                       ops::ArgsortKernel<paddle::platform::CPUPlace, double>,
                       ops::ArgsortKernel<paddle::platform::CPUPlace, int>,
                       ops::ArgsortKernel<paddle::platform::CPUPlace, int64_t>);
REGISTER_OP_CPU_KERNEL(
    argsort_grad, ops::ArgsortGradientKernel<paddle::platform::CPUPlace, float>,
    ops::ArgsortGradientKernel<paddle::platform::CPUPlace, double>,
    ops::ArgsortGradientKernel<paddle::platform::CPUPlace, int>,
    ops::ArgsortGradientKernel<paddle::platform::CPUPlace, int64_t>);
