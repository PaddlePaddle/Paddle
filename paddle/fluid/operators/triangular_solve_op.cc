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

#include "paddle/fluid/operators/triangular_solve_op.h"
#include "paddle/fluid/operators/solve_op.h"

namespace paddle {
namespace operators {

class TriangularSolveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "TriangularSolve");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "TriangularSolve");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "TriangularSolve");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_dims_n = x_dims.size();
    auto y_dims_n = y_dims.size();

    PADDLE_ENFORCE_GE(
        x_dims_n, 2, platform::errors::InvalidArgument(
                         "The input tensor X's dimensions of TriangularSolveOp "
                         "should be >= 2. But received X's "
                         "dimensions = %d, X's shape = [%s]",
                         x_dims.size(), x_dims));

    PADDLE_ENFORCE_GE(
        y_dims_n, 2, platform::errors::InvalidArgument(
                         "The input tensor Y's dimensions of TriangularSolveOp "
                         "should be >=2. But received Y's "
                         "dimensions = %d, Y's shape = [%s]",
                         y_dims.size(), y_dims));

    PADDLE_ENFORCE_EQ(x_dims[x_dims_n - 2], x_dims[x_dims_n - 1],
                      platform::errors::InvalidArgument(
                          "The inner-most 2 dimensions of Input(X) all should "
                          "be square matrices "
                          "But received X's shape[-2] = %d and shape[-1] = %d.",
                          x_dims[x_dims_n - 2], x_dims[x_dims_n - 1]));

    std::vector<int64_t> x_dims_vec = paddle::framework::vectorize(x_dims);
    std::vector<int64_t> y_dims_vec = paddle::framework::vectorize(y_dims);

    std::vector<int64_t> x_dims_vec_cut(x_dims_vec.begin(),
                                        x_dims_vec.end() - 2);
    std::vector<int64_t> y_dims_vec_cut(y_dims_vec.begin(),
                                        y_dims_vec.end() - 2);

    std::vector<int64_t> expand_batch_portion =
        get_broadcast_batch_portion(x_dims_vec_cut, y_dims_vec_cut);

    std::vector<int64_t> y_broadcast_dims({expand_batch_portion});
    y_broadcast_dims.insert(y_broadcast_dims.end(), {y_dims_vec[y_dims_n - 2],
                                                     y_dims_vec[y_dims_n - 1]});

    // dim of 'Out' is the same with 'Y' after broadcast
    ctx->SetOutputDim("Out", framework::make_ddim(y_broadcast_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

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
REGISTER_OPERATOR(triangular_solve, ops::TriangularSolveOp,
                  ops::TriangularSolveOpMaker,
                  ops::TriangularSolveOpInferVarType,
                  ops::TriangularSolveOpGradMaker<paddle::framework::OpDesc>,
                  ops::TriangularSolveOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(triangular_solve_grad, ops::TriangularSolveGradOp);

REGISTER_OP_CPU_KERNEL(
    triangular_solve,
    ops::TriangularSolveKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TriangularSolveKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    triangular_solve_grad,
    ops::TriangularSolveGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TriangularSolveGradKernel<paddle::platform::CPUDeviceContext, double>);
