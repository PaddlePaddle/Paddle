// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/conj_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class ConjOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of ConjOp should not be null."));
    PADDLE_ENFORCE_GE(ctx->Outputs("Out"), true,
                      platform::errors::InvalidArgument(
                          "Outputs(Out) of ConjOp should not be empty."));

    auto in_dims = ctx->GetInputDim("X");
    auto outs_names = ctx->Outputs("Out");

    ctx->SetOutputDim("Out", framework::make_ddim(in_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ConjOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of conj op.");
    AddOutput("Out", "(Tensor), The output tensor of conj op.");
    AddComment(R"DOC(
Conj Operator.

This operator is used to perform elementwise conjugate for input $X$.

)DOC");
  }
};

template <typename T>
class ConjOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("conj");
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("Out"), this->InputGrad("Out"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(conj, ops::ConjOp, ops::ConjOpMaker, ops::ConjOpInferVarType,
                  ops::ConjOpGradMaker<paddle::framework::OpDesc>,
                  ops::ConjOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(conj_grad, ops::ConjGradOp);

REGISTER_OP_CPU_KERNEL(conj, ops::MulKernel<paddle::platform::CPUDeviceContext,
                                            paddle::platform::complex64>,
                       ops::MulKernel<paddle::platform::CPUDeviceContext,
                                      paddle::platform::complex128>);
REGISTER_OP_CPU_KERNEL(conj_grad,
                       ops::MulGradKernel<paddle::platform::CPUDeviceContext,
                                          paddle::platform::complex64>,
                       ops::MulGradKernel<paddle::platform::CPUDeviceContext,
                                          paddle::platform::complex128>);