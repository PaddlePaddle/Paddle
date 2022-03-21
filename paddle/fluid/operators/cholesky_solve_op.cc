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
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class CholeskySolveOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(Solves a linear system of equations with a positive "
                "semidefinite matrix to be inverted given its Cholesky factor matrix uu."
                ")DOC");
    AddInput("X", "(Tensor) The input tensor, shape of (*,m,k)");
    AddInput("Y",
             "(Tensor) The input tensor, shape of (*,m,m) composed of upper or "
             "lower triangular Cholesky factor");
    AddOutput("Out", "(Tensor) The output tensor, shape same to X");
    AddAttr<bool>("upper",
                  "whether to consider the Cholesky factor "
                  "as a lower or upper triangular matrix")
        .SetDefault(false);
  }
};

class CholeskySolveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Y"), ctx.GetPlace());
  }
};

class CholeskySolveOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_type = ctx->GetInputType("Y", 0);
    auto data_type = ctx->GetInputDataType("Y", 0);

    ctx->SetOutputType("Out", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType("Out", data_type, framework::ALL_ELEMENTS);
  }
};

template <typename T>
class CholeskySolveOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("cholesky_solve_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput("Out", this->Output("Out"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};

class CholeskySolveGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "cholesky_solve");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "cholesky_solve");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "cholesky_solve");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "cholesky_solve");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(cholesky_solve, CholeskySolveInferShapeFunctor,
                            PD_INFER_META(phi::CholeskySolveInferMeta));

REGISTER_OPERATOR(cholesky_solve, ops::CholeskySolveOp,
                  ops::CholeskySolveOpMaker,
                  ops::CholeskySolveOpVarTypeInference,
                  ops::CholeskySolveOpGradMaker<paddle::framework::OpDesc>,
                  ops::CholeskySolveOpGradMaker<paddle::imperative::OpBase>,
                  CholeskySolveInferShapeFunctor);

REGISTER_OPERATOR(cholesky_solve_grad, ops::CholeskySolveGradOp);
