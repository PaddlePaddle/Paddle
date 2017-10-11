/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/mul_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class MulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of MulOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) of MulOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MulOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    int x_num_col_dims = ctx->Attrs().Get<int>("x_num_col_dims");
    int y_num_col_dims = ctx->Attrs().Get<int>("y_num_col_dims");

    PADDLE_ENFORCE_GT(
        x_dims.size(), x_num_col_dims,
        "The input tensor X's rank of MulOp should be larger than "
        "x_num_col_dims.");
    PADDLE_ENFORCE_GT(
        y_dims.size(), y_num_col_dims,
        "The input tensor Y's rank of MulOp should be larger than "
        "y_num_col_dims.");

    auto x_mat_dims = framework::flatten_to_2d(x_dims, x_num_col_dims);
    auto y_mat_dims = framework::flatten_to_2d(y_dims, y_num_col_dims);

    PADDLE_ENFORCE_EQ(
        x_mat_dims[1], y_mat_dims[0],
        "First matrix's width must be equal with second matrix's height.");
    ctx->SetOutputDim("Out", {x_mat_dims[0], y_mat_dims[1]});
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class MulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MulOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of mul op");
    AddInput("Y", "The second input of mul op");
    AddOutput("Out", "The output of mul op");
    AddAttr<int>(
        "x_num_col_dims",
        R"DOC(mul_op can take tensors with more than two dimensions as input `X`,
            in that case, tensors will be reshaped to a matrix. The matrix's first
            dimension(column length) will be the product of tensor's last
            `num_col_dims` dimensions, and the matrix's second dimension(row length)
            will be the product of tensor's first `rank - num_col_dims` dimensions.
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<int>(
        "y_num_col_dims",
        R"DOC(mul_op can take tensors with more than two dimensions as input `Y`,
             in that case, tensors will be reshaped to a matrix. Just like input `X`.
        )DOC")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddComment(R"DOC(
Mul operator is used to perform matrix multiplication for input X and Y.

The equation is:

    Out = X * Y

Both the input `X` and `Y` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD with input `X`.
)DOC");
  }
};

class MulOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    auto x_mat_dims =
        framework::flatten_to_2d(x_dims, Attr<int>("x_num_col_dims"));
    auto y_mat_dims =
        framework::flatten_to_2d(y_dims, Attr<int>("y_num_col_dims"));

    PADDLE_ENFORCE_EQ(
        x_mat_dims[0], out_dims[0],
        "The first dimension of Out@GRAD must equal to the first dimension of "
        "the first operand.");
    PADDLE_ENFORCE_EQ(
        y_mat_dims[1], out_dims[1],
        "The second dimension of Out@GRAD must equal to the second "
        "dimension of the second operand.");

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
REGISTER_OP(mul, ops::MulOp, ops::MulOpMaker, mul_grad, ops::MulOpGrad);
REGISTER_OP_CPU_KERNEL(mul, ops::MulKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(mul_grad,
                       ops::MulGradKernel<paddle::platform::CPUPlace, float>);
