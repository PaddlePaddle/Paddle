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

#include "paddle/operators/conv_shift_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class ConvShiftOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(y_dims.size(), 2, "Input(Y)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(x_dims[0], y_dims[0],
                      "The 1st dimension of Input(X) and Input(Y) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(y_dims[1] % 2, 1,
                      "The 2nd dimension of Input(Y) should be odd.")
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ConvShiftGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should be not null.");

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      auto x_dims = ctx->GetInputDim("X");
      ctx->SetOutputDim(x_grad_name, x_dims);
    }

    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(y_grad_name)) {
      auto y_dims = ctx->GetInputDim("Y");
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

class ConvShiftOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ConvShiftOpMaker(framework::OpProto* proto,
                   framework::OpAttrChecker* op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor, default Tensor<float>), a 2-D tensor with shape B x M, "
             "where B is the batch size and M is the data dimension.");
    AddInput("Y",
             "(Tensor, default Tensor<float>), a 2-D tensor with shape B x N, "
             "where B is the batch size and N is the data dimension. N must "
             "be odd.");
    AddOutput("Out",
              "(Tensor, default Tensor<float>), a 2-D tensor with shape B x M, "
              "i.e., the same shape as X.");
    AddComment(R"DOC(
ConvShift Operator.

A layer for circular convolution of two vectors,
as used in the Neural Turing Machine: https://arxiv.org/abs/1410.5401

The equation is:

  \f[
      Out[i] = \sum_{j=-(N-1)/2}^{(N-1)/2} X_{i+j} * Y_{j}
  \f]

where X's index is computed modulo M, and b's index is computed modulo N.

Both of the input `X` and `Y` can carry LoD (Level of Details) information.
However, the output only shares the LoD information with input `X`.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(conv_shift, ops::ConvShiftOp, ops::ConvShiftOpMaker,
            conv_shift_grad, ops::ConvShiftGradOp);
REGISTER_OP_CPU_KERNEL(conv_shift,
                       ops::ConvShiftKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    conv_shift_grad,
    ops::ConvShiftGradKernel<paddle::platform::CPUPlace, float>);
