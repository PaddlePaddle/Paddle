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
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class CumprodOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class CumprodOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of cumprod op.");
    AddOutput("Out", "(Tensor), The output tensor of cumprod op.");
    AddAttr<int>(
        "dim",
        "（int), The dim along which the input tensors will be cumproded");
    AddComment(
        R"DOC(Cumprod operator. Return the cumprod results of the input elements along the dim.
              For example, if input X is a tensor with rank 1 and N elements, the output will also be a tensor
              with rank 1 and N elements, and elements y[i] = x[0] * x[1] * x[2] *...* x[i] (0<=i<N))DOC");
  }
};

template <typename T>
class CumprodGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("cumprod_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Out", this->Output("Out"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class CumprodGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "CumprodGrad");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "CumprodGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "framework::GradVarName(\"Out\")",
                   "CumprodGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   "framework::GradVarName(\"X\")",
                   "CumprodGrad");
    ctx->ShareDim(framework::GradVarName("Out"), framework::GradVarName("X"));
    ctx->ShareLoD(framework::GradVarName("Out"), framework::GradVarName("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(cumprod,
                            CumprodInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(cumprod,
                  ops::CumprodOp,
                  ops::CumprodOpMaker,
                  ops::CumprodGradOpMaker<paddle::framework::OpDesc>,
                  ops::CumprodGradOpMaker<paddle::imperative::OpBase>,
                  CumprodInferShapeFunctor);

REGISTER_OPERATOR(cumprod_grad, ops::CumprodGradOp);
