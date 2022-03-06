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
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class CholeskyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class CholeskyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), The input tensor of cholesky op. Its shape should be "
             "[*, M, M] where * is zero or more batch dimensions, and matrices "
             "on the inner-most 2 dimensions all should be symmetric "
             "positive-definite.");
    AddOutput("Out",
              "(Tensor), The output tensor of cholesky op. It has the same "
              "shape as the input, and it is composed of upper-triangular or "
              "lower-triangular Cholesky factors of each of the individual "
              "matrices.");
    AddAttr<bool>("upper",
                  "(bool, default false), flag indicating whether to return "
                  "upper or lower triangular matrices. Default: False")
        .SetDefault(false);
    AddComment(R"DOC(
Cholesky Operator.

Computes the Cholesky decomposition of one symmetric positive-definite matrix
or batches of symmetric positive-definite matrices.

)DOC");
  }
};

class CholeskyGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "CholeskyGrad");
    OP_INOUT_CHECK(ctx->HasInputs(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "CholeskyGrad");
    auto dims = ctx->GetInputDim("Out");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, dims);
    }
  }
};

template <typename T>
class CholeskyGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(cholesky, CholeskyInferShapeFunctor,
                            PD_INFER_META(phi::CholeskyInferMeta));
REGISTER_OPERATOR(cholesky, ops::CholeskyOp, ops::CholeskyOpMaker,
                  ops::CholeskyGradOpMaker<paddle::framework::OpDesc>,
                  ops::CholeskyGradOpMaker<paddle::imperative::OpBase>,
                  CholeskyInferShapeFunctor);
REGISTER_OPERATOR(cholesky_grad, ops::CholeskyGradOp);
