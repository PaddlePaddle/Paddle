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

#include "paddle/operators/softmax_op.h"

namespace paddle {
namespace operators {

class SoftmaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SoftmaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"),
                   "Output(Y) of SoftmaxOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(x_dims.size() == 2UL,
                   "The input of softmax op must be a matrix.");
    ctx->SetOutputDim("Y", x_dims);
  }
};

class SoftmaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SoftmaxOpMaker(framework::OpProto* proto,
                 framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "The input tensor of softmax. "
             "2-D with shape [batch_size, input_feature_dimensions].");
    AddOutput("Y", "The normalized values with the same shape as X.");
    AddComment(R"DOC(
Softmax Operator.

The input of the softmax operator is a 2-D tensor with shape N x K (N is the
batch_size, K is the dimension of input feature). The output tensor has the
same shape as the input tensor.

For each row of the input tensor, the softmax operator squashes the
K-dimensional vector of arbitrary real values to a K-dimensional vector of real
values in the range [0, 1] that add up to 1.
It computes the exponential of the given dimension and the sum of exponential
values of all the other dimensions in the K-dimensional vector input.
Then the ratio of the exponential of the given dimension and the sum of
exponential values of all the other dimensions is the output of the softmax
operator.

For each row `i` and each column `j` in input X, we have:
    $$Y[i, j] = \frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])}$$

)DOC");
  }
};

class SoftmaxOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) should be not null.");
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Y"),
                      ctx->GetInputDim(framework::GradVarName("Y")),
                      "Input(Y) and its gradients should have a same shape.");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP(softmax, ops::SoftmaxOp, ops::SoftmaxOpMaker, softmax_grad,
            ops::SoftmaxOpGrad);
REGISTER_OP_CPU_KERNEL(softmax,
                       ops::SoftmaxKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    softmax_grad, ops::SoftmaxGradKernel<paddle::platform::CPUPlace, float>);
