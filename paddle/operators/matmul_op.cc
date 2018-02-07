/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/matmul_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class MatMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE(context->HasInput("X"),
                   "Input(X) of MatMulOp should not be null.");
    PADDLE_ENFORCE(context->HasInput("Y"),
                   "Input(Y) of MatMulOp should not be null.");
    PADDLE_ENFORCE(context->HasOutput("Out"),
                   "Output(Out) of MatMulOp should not be null.");

    auto dim_x = context->GetInputDim("X");
    auto dim_y = context->GetInputDim("Y");
    bool transpose_x = context->Attrs().Get<bool>("transpose_X");
    bool transpose_y = context->Attrs().Get<bool>("transpose_Y");

    int x_num_col_dims = context->Attrs().Get<int>("x_num_col_dims");
    int y_num_col_dims = context->Attrs().Get<int>("y_num_col_dims");

    PADDLE_ENFORCE_GE(dim_x.size(), 1,
                      "Input tensor X must be at least 1-dimensional.");
    PADDLE_ENFORCE_GE(dim_y.size(), 1,
                      "Input tensor Y must be at least 1-dimensional.");

    std::vector<int64_t> dim_out;
    if (x_num_col_dims == 0 && x_num_col_dims == 0) {
      std::vector<int64_t> out_dim;
      int64_t batch_count = 1;
      if (dim_x.size() > 3) {
        PADDLE_ENFORCE_EQ(
            dim_y.size(), dim_x.size(),
            "The dimensions of X and Y must be the same, and both of "
            "them should be %d-dimensional.",
            dim_x.size());

        // The first rank-2 dimensions are accumulated on the batch_count,
        // and the last two dimensions are used for matrix multiplication.
        for (int j = 0; j < dim_x.size() - 2; ++j) {
          PADDLE_ENFORCE_EQ(dim_y[j], dim_x[j],
                            "The %d-th dimension of X and Y must be the same.",
                            j);
          out_dim.push_back(dim_x[j]);
          batch_count *= dim_x[j];
        }
      }

      int M = 0, N = 0, KX = 0, KY = 0, batchCountX = 0, batchCountY = 0;
      bool remove_initial_dim = false, remove_final_dim = false;

      switch (dim_x.size()) {
        case 1:
          if (transpose_x) {
            M = dim_x[0];
            KX = 1;
          } else {
            M = 1;
            KX = dim_x[0];
            remove_initial_dim = true;
          }
          break;
        case 2:
          M = transpose_x ? dim_x[1] : dim_x[0];
          KX = transpose_x ? dim_x[0] : dim_x[1];
          break;
        case 3:
          batchCountX = dim_x[0];
          M = transpose_x ? dim_x[2] : dim_x[1];
          KX = transpose_x ? dim_x[1] : dim_x[2];
          break;
        default:
          batchCountX = batch_count;
          size_t mat_s = dim_x.size() - 2;
          M = transpose_x ? dim_x[mat_s + 1] : dim_x[mat_s];
          KX = transpose_x ? dim_x[mat_s] : dim_x[mat_s + 1];
          break;
      }

      switch (dim_y.size()) {
        case 1:
          if (transpose_y) {
            N = dim_y[0];
            KY = 1;
          } else {
            N = 1;
            KY = dim_y[0];
            remove_final_dim = true;
          }
          break;
        case 2:
          KY = transpose_y ? dim_y[1] : dim_y[0];
          N = transpose_y ? dim_y[0] : dim_y[1];
          break;
        case 3:
          batchCountY = dim_y[0];
          KY = transpose_y ? dim_y[2] : dim_y[1];
          N = transpose_y ? dim_y[1] : dim_y[2];
          break;
        default:
          batchCountY = batch_count;
          size_t mat_s = dim_y.size() - 2;
          KY = transpose_y ? dim_y[mat_s + 1] : dim_y[mat_s];
          N = transpose_y ? dim_y[mat_s] : dim_y[mat_s + 1];
      }

      PADDLE_ENFORCE_EQ(
          KX, KY,
          "First matrix's width must be equal with second matrix's height.");
      if (batchCountX && batchCountY) {
        PADDLE_ENFORCE_EQ(
            batchCountX, batchCountY,
            "When Input(X) and Input(Y) are both three dimensional, they "
            "must have the same batch dimension.");
      }
      int batchCount = std::max(batchCountX, batchCountY);

      if (batchCount) {
        if (dim_x.size() > 3) {
          dim_out.insert(dim_out.begin(), out_dim.begin(), out_dim.end());
        } else {
          dim_out.push_back(batchCount);
        }
      }
      if (!remove_initial_dim) {
        dim_out.push_back(M);
      }
      if (!remove_final_dim) {
        dim_out.push_back(N);
      }
      if (dim_out.size() == 0) {
        // We don't support 0-dimensional Tensors (scalars), so instead
        // treat the output as a Tensor of shape (1, ) in this case.
        dim_out.push_back(1);
      }
    } else {
      if (x_num_col_dims == 0) {
        x_num_col_dims = 1;
      }
      if (y_num_col_dims == 0) {
        y_num_col_dims = 1;
      }
      PADDLE_ENFORCE_GT(
          dim_x.size(), x_num_col_dims,
          "The input tensor X's rank of MulOp should be larger than "
          "x_num_col_dims.");
      PADDLE_ENFORCE_GT(
          dim_x.size(), y_num_col_dims,
          "The input tensor Y's rank of MulOp should be larger than "
          "y_num_col_dims.");

      auto x_mat_dims = framework::flatten_to_2d(dim_x, x_num_col_dims);
      auto y_mat_dims = framework::flatten_to_2d(dim_y, y_num_col_dims);

      PADDLE_ENFORCE_EQ(
          x_mat_dims[1], y_mat_dims[0],
          "First matrix's width must be equal with second matrix's height.");

      dim_out.reserve(
          static_cast<size_t>(x_num_col_dims + dim_y.size() - y_num_col_dims));

      for (int i = 0; i < x_num_col_dims; ++i) {
        dim_out.push_back(dim_x[i]);
      }

      for (int i = y_num_col_dims; i < dim_y.size(); ++i) {
        dim_out.push_back(dim_y[i]);
      }
    }
    context->SetOutputDim("Out", framework::make_ddim(dim_out));
    context->ShareLoD("X", /*->*/ "Out");
  }
};

class MatMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MatMulOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of MatMul op");
    AddInput("Y", "The second input of MatMul op");
    AddOutput("Out", "The output of MatMul op");
    AddAttr<int>(
        "x_num_col_dims",
        R"DOC((int, default 0), The matmul_op can take tensors with more than two
              dimensions as its inputs. If the input $X$ is a tensor with more
              than two dimensions, $X$ will be flattened into a two-dimensional
              matrix first. The flattening rule is: the first `num_col_dims`
              will be flattened to form the first dimension of the final matrix
              (the height of the matrix), and the rest `rank(X) - num_col_dims`
              dimensions are flattened to form the second dimension of the final
              matrix (the width of the matrix). As a result, height of the
              flattened matrix is equal to the product of $X$'s first
              `x_num_col_dims` dimensions' sizes, and width of the flattened
              matrix is equal to the product of $X$'s last `rank(x) - num_col_dims`
              dimensions' size. For example, suppose $X$ is a 6-dimensional
              tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3.
              Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] =
              [24, 30]. The default value 0 indicates the input is a 2-D Matrix.
        )DOC")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddAttr<int>(
        "y_num_col_dims",
        R"DOC((int, default 0), The matmul_op can take tensors with more than
              two, dimensions as its inputs. If the input $Y$ is a tensor with
              more than two dimensions, $Y$ will be flattened into a
              two-dimensional matrix first. The attribute `y_num_col_dims`
              determines how $Y$ is flattened.
              See comments of `x_num_col_dims` for more details.
        )DOC")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddAttr<bool>("transpose_X",
                  R"DOC(If true, use the transpose of `X`.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>("transpose_Y",
                  R"DOC(If true, use the transpose of `Y`.
        )DOC")
        .SetDefault(false);
    AddComment(R"DOC(
MatMul Operator.


This operator is used to perform (batched) matrix multiplication
over the last two dimensions of the input tensors `X` and `Y`.

If a transpose flag is specified, the last two dimensions of the
tensor are transposed. If the tensor is rank-1 of shape [D], then
for `X` it is treated as [1, D] in nontransposed form and as [D, 1]
in transposed form, whereas for `Y` it is the opposite: It is treated
as [D, 1] in nontransposed form and as [1, D] in transposed form.

Examples without transpose:
- X: [K], Y: [K] => Out: [1]
- X: [K], Y: [K, N] => Out: [N]
- X: [B, M, K], Y: [K] => Out: [B, M]
- X: [M, K], Y: [B, K, N] => Out: [B, M, N]
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, N]
- X: [B, ..., M, K], Y: [B, ..., K, N] => Out: [B, ..., M, N]

The behavior is designed to be similar to the `numpy.matmul` function.
The differences are:
- When the rank of the input data is less than or equal to 3, it
  is similar to the `numpy.matmul` function.
- When the rank of the input is greater than 3, the rank of X and
  Y must be equal, and the first `rank - 2` dimensions must be equal.
- We add `transpose_X` and `transpose_Y` flags.

Both the input `X` and `Y` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input `X`.

)DOC");
  }
};

class MatMulOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE(context->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(context->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(context->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = context->GetInputDim("X");
    auto y_dims = context->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (context->HasOutput(x_grad_name)) {
      context->SetOutputDim(x_grad_name, x_dims);
    }
    if (context->HasOutput(y_grad_name)) {
      context->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(matmul, ops::MatMulOp, ops::MatMulOpMaker, matmul_grad,
            ops::MatMulOpGrad);
REGISTER_OP_CPU_KERNEL(
    matmul, ops::MatMulKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    matmul_grad,
    ops::MatMulGradKernel<paddle::platform::CPUDeviceContext, float>);
