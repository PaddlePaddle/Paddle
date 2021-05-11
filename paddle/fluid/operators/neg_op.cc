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

#include "paddle/fluid/operators/neg_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

class NegOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of neg op.");
    AddOutput("Out", "(Tensor), The output tensor of neg op.");
    AddComment(R"DOC(
Neg Operator.

This operator is used to perform elementwise neg for input $X$.
$$out = |x|$$

)DOC");
  }
};

template <typename T>
class NegGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("neg_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

class NegOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "neg");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "neg");

    auto in_dims = ctx->GetInputDim("X");

    ctx->SetOutputDim("Out", in_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class NegGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
		   "Out@Grad", "NegGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@Grad", "AbsGrad");

    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), dout_dims);
    ctx->ShareLoD(framework::GradVarName("Out"), framework::GradVarName("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(neg, ops::NegOp, ops::NegOpMaker,
                  ops::NegGradMaker<paddle::framework::OpDesc>,
                  ops::NegGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(neg_grad, ops::NegGradOp);

REGISTER_OP_CPU_KERNEL(neg,
    ops::NegKernel<paddle::platform::CPUDeviceContext, float>,
    ops::NegKernel<paddle::platform::CPUDeviceContext, double>,
    ops::NegKernel<paddle::platform::CPUDeviceContext, int>,
    ops::NegKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::NegKernel<paddle::platform::CPUDeviceContext,
                   paddle::platform::float16>);

REGISTER_OP_CPU_KERNEL(neg_grad,
    ops::NegGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::NegGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::NegGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::NegGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::NegGradKernel<paddle::platform::CPUDeviceContext,
                       paddle::platform::float16>);
