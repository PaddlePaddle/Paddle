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

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

class MultiDotOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensors of multi_dot operator.").AsDuplicable();
    AddOutput("Out", "The output tensor of multi_dot operator");
    AddComment(R"DOC(
Compute the dot product of two or more arrays in a single function call, while automatically selecting the fastest evaluation order.

multi_dot chains MatMul and uses optimal parenthesization of the matrices [1] [2]. Depending on the shapes of the matrices, this can speed up the multiplication a lot.

If the first argument is 1-D it is treated as a row vector. If the last argument is 1-D it is treated as a column vector. The other arguments must be 2-D.
      )DOC");
  }
};

class MultiDotOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class MultiDotOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "multi_dot");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "multi_dot");

    auto in_x = "X";
    auto out_x_g_n = framework::GradVarName(in_x);
    auto ins_dims = ctx->GetInputsDim(in_x);
    ctx->SetOutputsDim(out_x_g_n, ins_dims);
    ctx->ShareAllLoD(in_x, out_x_g_n);
  }
};

template <typename T>
class MultiDotOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("multi_dot_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X", false));
  }
};
template <typename T>
class MultiDotOpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("multi_dot");
    grad_op->SetInput("X", this->Input(("X")));
    grad_op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    grad_op->SetOutput("DDx", this->OutputGrad(framework::GradVarName("X")));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(multi_dot,
                            MultiDotInferShapeFunctor,
                            PD_INFER_META(phi::MultiDotInferMeta));

REGISTER_OPERATOR(multi_dot,
                  ops::MultiDotOp,
                  ops::MultiDotOpMaker,
                  ops::MultiDotOpGradMaker<paddle::framework::OpDesc>,
                  ops::MultiDotOpGradMaker<paddle::imperative::OpBase>,
                  MultiDotInferShapeFunctor);

REGISTER_OPERATOR(multi_dot_grad,
                  ops::MultiDotOpGrad,
                  ops::MultiDotOpDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::MultiDotOpDoubleGradMaker<paddle::imperative::OpBase>);
