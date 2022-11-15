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

#include "paddle/fluid/operators/squared_l2_distance_op.h"

#include <memory>

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"

namespace paddle {
namespace operators {

class SquaredL2DistanceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SquaredL2DistanceOp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "SquaredL2DistanceOp");
    OP_INOUT_CHECK(ctx->HasOutput("sub_result"),
                   "Output",
                   "sub_result",
                   "SquaredL2DistanceOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Out"), "Output", "Out", "SquaredL2DistanceOp");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(phi::arity(x_dims),
                      phi::arity(y_dims),
                      platform::errors::InvalidArgument(
                          "Input(X) and Input(X) of SquaredL2DistanceOp should "
                          "have same dimensions. "
                          "But received X's shape = [%s] and Y's shape = [%s], "
                          "the dimensions are %d and %d respectively",
                          x_dims,
                          y_dims,
                          phi::arity(x_dims),
                          phi::arity(y_dims)));

    int rank = phi::arity(x_dims);
    PADDLE_ENFORCE_GE(
        rank,
        2,
        platform::errors::InvalidArgument(
            "Input dimensions of SquaredL2DistanceOp should be at least 2."
            "But received shape = [%s] and dimension is %d.",
            x_dims,
            rank));
    bool check = true;
    if ((!ctx->IsRuntime()) &&
        (phi::product(x_dims) <= 0 || phi::product(y_dims) <= 0)) {
      check = false;
    }
    if (check) {
      PADDLE_ENFORCE_EQ(
          product(x_dims) / x_dims[0],
          product(y_dims) / y_dims[0],
          platform::errors::InvalidArgument(
              "Input(X) and Input(Y) of SquaredL2DistanceOp should "
              "have same dimensions."
              "But received X's shape = [%s] and Y's shape = [%s]"
              ", the products are %d and %d respectively",
              x_dims,
              y_dims,
              product(x_dims) / x_dims[0],
              product(y_dims) / y_dims[0]));
    }
    check = true;
    if ((!ctx->IsRuntime()) && (y_dims[0] <= 0 || x_dims[0] <= 0)) {
      check = false;
    }
    if (check) {
      PADDLE_ENFORCE_EQ(
          y_dims[0] == 1 || y_dims[0] == x_dims[0],
          true,
          platform::errors::InvalidArgument(
              "First dimension of Input(Y) of SquaredL2DistanceOp "
              "must be equal to 1 or to first dimension of Input(X)."
              "But received X's shape = [%s] and Y's shape = [%s],"
              "the first dimensions are %d and %d respectively",
              x_dims,
              y_dims,
              x_dims[0],
              y_dims[0]));
    }
    ctx->SetOutputDim("sub_result", {x_dims[0], product(x_dims) / x_dims[0]});
    ctx->SetOutputDim("Out", {x_dims[0], 1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SquaredL2DistanceGradOpNoBufferVarsInferer,
                                    "X",
                                    "Y");

template <typename T>
class SquaredL2DistanceGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("squared_l2_distance_grad");

    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("sub_result", this->Output("sub_result"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));

    op->SetAttrMap(this->Attrs());
  }
};

class SquaredL2DistanceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input of SquaredL2DistanceOp.");
    AddInput("Y", "(Tensor) Target of SquaredL2DistanceOp.");
    AddOutput("sub_result",
              "(Tensor) Buffering subtraction result which "
              "will be reused in backward.")
        .AsIntermediate();
    AddOutput("Out", "(Tensor) Squared l2 distance between input and target.");
    AddComment(R"DOC(
SquaredL2Distance operator

This operator will cacluate the squared L2 distance for the input and
the target. Number of distance value will be equal to the first dimension
of input. First dimension of the target could be equal to the input or to 1.
If the first dimension of target is 1, the operator will broadcast target's
first dimension to input's first dimension. During backward propagation,
the user can decide whether to calculate the gradient of the input or
the target or both.

Both the input X and Y can carry the LoD (Level of Details) information.
However, the output only shares the LoD information with input X.
    )DOC");
  }
};

class SquaredL2DistanceGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("sub_result"),
                   "Input",
                   "sub_result",
                   "SquaredL2DistanceGradOp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "SquaredL2DistanceGradOp");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          out_dims[0],
          x_dims[0],
          platform::errors::InvalidArgument(
              "First dimension of output gradient and Input(X) "
              "of SquaredL2DistanceGradOp must be equal "
              "But received X's shape = [%s] and grad's shape = [%s], "
              "the first dimensions are %d and %d respectively",
              x_dims,
              out_dims,
              x_dims[0],
              out_dims[0]));
      PADDLE_ENFORCE_EQ(out_dims[1],
                        1,
                        platform::errors::InvalidArgument(
                            "Second dimension of output gradient of "
                            "SquaredL2DistanceGradOp must be 1. "
                            "But received grad's shape = [%s], "
                            "with second dimension %d",
                            out_dims,
                            out_dims[1]));
    }
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) ctx->SetOutputDim(x_grad_name, x_dims);
    if (ctx->HasOutput(y_grad_name)) ctx->SetOutputDim(y_grad_name, y_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "sub_result"),
        ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    squared_l2_distance,
    ops::SquaredL2DistanceOp,
    ops::SquaredL2DistanceOpMaker,
    ops::SquaredL2DistanceGradOpMaker<paddle::framework::OpDesc>,
    ops::SquaredL2DistanceGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(squared_l2_distance_grad,
                  ops::SquaredL2DistanceGradOp,
                  ops::SquaredL2DistanceGradOpNoBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(squared_l2_distance,
                       ops::SquaredL2DistanceKernel<phi::CPUContext, float>);
REGISTER_OP_CPU_KERNEL(
    squared_l2_distance_grad,
    ops::SquaredL2DistanceGradKernel<phi::CPUContext, float>);
