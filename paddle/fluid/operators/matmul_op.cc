/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/matmul_op.h"
#include <algorithm>
#include <vector>

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

    int x_num_col_dims = context->Attrs().Get<int>("x_num_col_dims");
    int y_num_col_dims = context->Attrs().Get<int>("y_num_col_dims");

    auto mat_dim_x = math::GetMatDim(GetXDim(dim_x), x_num_col_dims,
                                     context->Attrs().Get<bool>("transpose_X"));
    auto mat_dim_y = math::GetMatDim(GetYDim(dim_y), y_num_col_dims,
                                     context->Attrs().Get<bool>("transpose_Y"));

    // Check
    PADDLE_ENFORCE_EQ(mat_dim_x.width_, mat_dim_y.height_);
    PADDLE_ENFORCE(mat_dim_x.batch_size_ == mat_dim_y.batch_size_ ||
                   mat_dim_x.batch_size_ == 0 || mat_dim_y.batch_size_ == 0);

    // Calc out dim
    std::vector<int64_t> dim_out;
    if (x_num_col_dims == 0 && y_num_col_dims == 0) {
      if (mat_dim_x.batch_size_ != 0) {
        dim_out = framework::vectorize(dim_x);
        dim_out[dim_out.size() - 2] = mat_dim_x.height_;
        dim_out[dim_out.size() - 1] = mat_dim_y.width_;
      } else if (mat_dim_y.batch_size_ != 0) {
        dim_out = framework::vectorize(dim_y);
        dim_out[dim_out.size() - 2] = mat_dim_x.height_;
        dim_out[dim_out.size() - 1] = mat_dim_y.width_;
      } else {
        dim_out = {mat_dim_x.height_, mat_dim_y.width_};
      }

      if (dim_x.size() == 1 && dim_out[dim_out.size() - 2] == 1) {
        std::swap(dim_out[dim_out.size() - 2], dim_out[dim_out.size() - 1]);
        dim_out.resize(dim_out.size() - 1);
      }

      if (dim_y.size() == 1 && dim_out[dim_out.size() - 1] == 1) {
        dim_out.resize(dim_out.size() - 1);
      }

      if (dim_out.empty()) {
        dim_out = {1};
      }
    } else {
      // They should be both > 0
      PADDLE_ENFORCE_GT(x_num_col_dims, 0);
      PADDLE_ENFORCE_GT(y_num_col_dims, 0);
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
    AddAttr<bool>("transpose_X",
                  R"DOC(If true, use the transpose of `X`.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>("transpose_Y",
                  R"DOC(If true, use the transpose of `Y`.
        )DOC")
        .SetDefault(false);
    AddAttr<int>("x_num_col_dims",
                 R"DOC(The mul_op can take tensors with more than two
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
              [24, 30].
        )DOC")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddAttr<int>("y_num_col_dims",
                 R"DOC(The mul_op can take tensors with more than two,
              dimensions as its inputs. If the input $Y$ is a tensor with more
              than two dimensions, $Y$ will be flattened into a two-dimensional
              matrix first. The attribute `y_num_col_dims` determines how $Y$ is
              flattened. See comments of `x_num_col_dims` for more details.
        )DOC")
        .SetDefault(0)
        .EqualGreaterThan(0);
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
REGISTER_OPERATOR(matmul, ops::MatMulOp, ops::MatMulOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(matmul_grad, ops::MatMulOpGrad);
REGISTER_OP_CPU_KERNEL(
    matmul, ops::MatMulKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MatMulKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MatMulKernel<paddle::platform::CPUDeviceContext,
                      paddle::platform::float16>);
REGISTER_OP_CPU_KERNEL(
    matmul_grad,
    ops::MatMulGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MatMulGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MatMulGradKernel<paddle::platform::CPUDeviceContext,
                          paddle::platform::float16>);
