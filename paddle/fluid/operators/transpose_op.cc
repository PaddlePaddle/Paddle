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

#include "paddle/fluid/operators/transpose_op.h"
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class TransposeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    std::vector<int> axis = ctx->Attrs().Get<std::vector<int>>("axis");
    size_t x_rank = x_dims.size();
    size_t axis_size = axis.size();

    PADDLE_ENFORCE_EQ(x_rank, axis_size,
                      "The input tensor's rank(%d) "
                      "should be equal to the axis's size(%d)",
                      x_rank, axis_size);

    std::vector<int> count(axis_size, 0);
    for (size_t i = 0; i < axis_size; i++) {
      PADDLE_ENFORCE(
          axis[i] < static_cast<int>(axis_size) && ++count[axis[i]] == 1,
          "Each element of Attribute axis should be a unique value "
          "range from 0 to (dims - 1), "
          "where the dims is the axis's size");
    }

    framework::DDim out_dims(x_dims);
    for (size_t i = 0; i < axis_size; i++) {
      out_dims[i] = x_dims[axis[i]];
    }
    ctx->SetOutputDim("Out", out_dims);
  }
};

class TransposeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor, tensors with rank up to 6 are supported.");
    AddOutput("Out", "(Tensor)The output tensor.");
    AddAttr<std::vector<int>>(
        "axis",
        "(vector<int>) A list of values, and the size of the list should be "
        "the same with the input tensor rank. This operator permutes the input "
        "tensor's axes according to the values given.");
    AddComment(R"DOC(
Transpose Operator.

The input tensor will be permuted according to the axes given.
The behavior of this operator is similar to how `numpy.transpose` works.

- suppose the input `X` is a 2-D tensor:
    $$
    X = \begin{pmatrix}
    0 &1 &2 \\
    3 &4 &5
    \end{pmatrix}$$

    the given `axes` is: $[1, 0]$, and $Y$ = transpose($X$, axis)

    then the output $Y$ is:

    $$
    Y = \begin{pmatrix}
         0 &3 \\
         1 &4  \\
         2 &5
    \end{pmatrix}$$

- Given a input tensor with shape $(N, C, H, W)$ and the `axes` is 
$[0, 2, 3, 1]$, then shape of the output tensor will be: $(N, H, W, C)$.

)DOC");
  }
};

class TransposeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(transpose, ops::TransposeOp, ops::TransposeOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(transpose_grad, ops::TransposeOpGrad);
REGISTER_OP_CPU_KERNEL(
    transpose, ops::TransposeKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    transpose_grad,
    ops::TransposeGradKernel<paddle::platform::CPUDeviceContext, float>);
