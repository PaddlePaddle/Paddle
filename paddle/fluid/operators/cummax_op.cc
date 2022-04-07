// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

class CummaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class CummaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of cummax operator");
    AddOutput("Out", "(Tensor), The output tensor of cummax operator");
    AddOutput("Indices", "(Tensor), The indices of cummax elements");
    AddAttr<int>("axis",
                 "The dimension of accumulate along, -1 mean the last"
                 "dimension [default -1]")
      .SetDefault(-1);
    AddComment(R"DOC(
Return the cumulative maximum elements and indices of input tensor along a given
axis. For example, if input X is a tensor, the output will also be a the same
shape tensor, and elements y[i] = max(x[0], x[1], ..., x[i]).
    )DOC");
  }
};

template <typename T>
class CummaxGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("cummax_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Indices", this->Output("Indices"));
    grad_op->SetInput(framework::GradVarName("Indices"),
                      this->OutputGrad("Indices"));
    grad_op->SetOutput(framework::GradVarName("X"),
                       this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class CummaxGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input",
                   "X", "CummaxGrad");
    OP_INOUT_CHECK(ctx->HasInput("Indices"),
                   "Input", "Indices", "CummaxGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Indices")),
                   "Input",
                   "framework::GradVarName(\"Indices\")",
                   "CummaxGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   "framework::GradVarName(\"X\")",
                   "CummaxGrad");
    ctx->ShareDim(framework::GradVarName("Indices"),
                  framework::GradVarName("X"));
    ctx->ShareLoD(framework::GradVarName("Indices"),
                  framework::GradVarName("X"));  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(cummax, CummaxInferShapeFunctor,
                            PD_INFER_META(phi::UnchangedInferMeta));

REGISTER_OPERATOR(cummax, ops::CummaxOp, ops::CummaxOpMaker,
                  ops::CummaxGradOpMaker<paddle::framework::OpDesc>,
                  ops::CummaxGradOpMaker<paddle::imperative::OpBase>,
                  CummaxInferShapeFunctor);

REGISTER_OPERATOR(cummax_grad, ops::CummaxGradOp);
