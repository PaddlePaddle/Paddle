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

class TriangularSolveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class TriangularSolveOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), The first input tensor of triangular solve op, which "
             "is the triangular coefficient matrix.");
    AddInput("Y",
             "(Tensor), The second input tensor of triangular solve op, which "
             "is multiple right-hand.");
    AddOutput("Out", "(Tensor), The solution tensor of triangular solve op.");
    AddAttr<bool>("upper",
                  "whether to solve the upper-triangular or the "
                  "lower-triangular system of equations")
        .SetDefault(true);
    AddAttr<bool>("transpose", "whether X should be transposed firstly.")
        .SetDefault(false);
    AddAttr<bool>("unitriangular", "whether X is unit triangular.")
        .SetDefault(false);
    AddComment(R"DOC(
          Triangular Solve Operator.
          This operator is used to computes the solution of equations with a triangular coefficient matrix.

          The equation is:
          $$Out = X^-1 * Y$$
)DOC");
  }
};

class TriangularSolveOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

class TriangularSolveGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "triangular_solve");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "triangular_solve");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "triangular_solve");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "triangular_solve");

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

template <typename T>
class TriangularSolveOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("triangular_solve_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput("Out", this->Output("Out"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(triangular_solve, TriangularSolveInferShapeFunctor,
                            PD_INFER_META(phi::TriangularSolveInferMeta));

REGISTER_OPERATOR(triangular_solve, ops::TriangularSolveOp,
                  ops::TriangularSolveOpMaker,
                  ops::TriangularSolveOpInferVarType,
                  ops::TriangularSolveOpGradMaker<paddle::framework::OpDesc>,
                  ops::TriangularSolveOpGradMaker<paddle::imperative::OpBase>,
                  TriangularSolveInferShapeFunctor);

REGISTER_OPERATOR(triangular_solve_grad, ops::TriangularSolveGradOp);
