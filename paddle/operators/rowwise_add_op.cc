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

#include "paddle/operators/rowwise_add_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class RowwiseAddOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContextBase &ctx) const override {
    PADDLE_ENFORCE(ctx.HasInput("X"),
                   "Input(X) of RowwiseAddOp should not be null.");
    PADDLE_ENFORCE(ctx.HasInput("b"),
                   "Input(b) of RowwiseAddOp should not be null.");
    PADDLE_ENFORCE(ctx.HasOutput("Out"),
                   "Output(Out) of RowwiseAddOp should not be null.");

    auto x_dims = ctx.GetInputDim("X");
    auto b_dims = ctx.GetInputDim("b");
    PADDLE_ENFORCE_GT(
        x_dims.size(), b_dims.size(),
        "The rank of input `X` must be larger than the one of input `b`.");

    int num_col_dims = x_dims.size() - b_dims.size();

    PADDLE_ENFORCE_EQ(
        framework::slice_ddim(x_dims, num_col_dims, x_dims.size()), b_dims,
        "The width of two operands must be same");
    PADDLE_ENFORCE_EQ(ctx.Outputs("Out").size(), 1,
                      "The output size must be 1");
    ctx.SetOutputDim("Out", x_dims);
  }
};

class RowwiseAddOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RowwiseAddOpMaker(framework::OpProto *proto,
                    framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The left input of row-wise add op, must be matrix");
    AddInput("b", "The right input of row-wise add op, must be vector");
    AddOutput("Out", "The output of row-wise add op");
    AddComment(R"DOC(Row-wise Add operator

for i in xrange(X.shape[0]):
  Out = X[i] + b
)DOC");
  }
};
class RowwiseAddGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContextBase &ctx) const override {
    PADDLE_ENFORCE(ctx.HasInput("X"), "X should not be null");
    PADDLE_ENFORCE(ctx.HasInput("b"), "b should not be null");
    PADDLE_ENFORCE(ctx.HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = ctx.GetInputDim("X");
    auto b_dims = ctx.GetInputDim("b");
    PADDLE_ENFORCE_GT(
        x_dims.size(), b_dims.size(),
        "The rank of input `X` must be larger than the one of input `b`.");

    int64_t num_col_dims = x_dims.size() - b_dims.size();
    PADDLE_ENFORCE_EQ(
        framework::slice_ddim(x_dims, num_col_dims, x_dims.size()), b_dims,
        "The width of two operands must be same");
    auto x_grad_name = framework::GradVarName("X");
    auto b_grad_name = framework::GradVarName("b");
    if (ctx.HasOutput(x_grad_name)) {
      ctx.SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx.HasOutput(b_grad_name)) {
      ctx.SetOutputDim(b_grad_name, b_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(rowwise_add, ops::RowwiseAddOp, ops::RowwiseAddOpMaker,
            rowwise_add_grad, ops::RowwiseAddGradOp);
REGISTER_OP_CPU_KERNEL(
    rowwise_add, ops::RowwiseAddKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    rowwise_add_grad,
    ops::RowwiseAddGradKernel<paddle::platform::CPUPlace, float>);
