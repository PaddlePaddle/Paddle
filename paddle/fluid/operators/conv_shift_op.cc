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

#include "paddle/fluid/operators/conv_shift_op.h"

#include <memory>

#include "paddle/fluid/framework/eigen.h"

namespace paddle {
namespace operators {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

class ConvShiftOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ConvShiftOp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ConvShiftOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ConvShiftOp");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_EQ(
        x_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "Input(X)'s dimensions of ConvShiftOp should be 2. "
            "But received X's shape = [%s] and the dimension is %d.",
            x_dims,
            x_dims.size()));
    PADDLE_ENFORCE_EQ(
        y_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "Input(Y)'s dimensions of ConvShiftOp should be 2. "
            "But received Y's shape = [%s] and the dimension is %d.",
            y_dims,
            y_dims.size()));
    if (ctx->IsRuntime() || (x_dims[0] > 0 && y_dims[0] > 0))
      PADDLE_ENFORCE_EQ(
          x_dims[0],
          y_dims[0],
          platform::errors::InvalidArgument(
              "The first dimension of Input(X) and Input(Y) of ConvShiftOp "
              "should be equal. "
              "But received X's shape = [%s], Y's shape = [%s], "
              "and the first dimensions are %d and %d respectively.",
              x_dims,
              y_dims,
              x_dims[0],
              y_dims[0]));
    if (ctx->IsRuntime() || y_dims[1] > 0)
      PADDLE_ENFORCE_EQ(
          y_dims[1] % 2,
          1,
          platform::errors::InvalidArgument(
              "The second dimension of Input(Y) of ConvShiftOp should be odd."
              "But received Y's shape = [%s] and the second dimension is %d.",
              y_dims,
              y_dims[1]));
    if (ctx->IsRuntime() || (x_dims[1] > 0 && y_dims[1] > 0))
      PADDLE_ENFORCE_LE(
          y_dims[1],
          x_dims[1],
          platform::errors::InvalidArgument(
              "The second dimension of Input(Y) of ConvShiftOp should be less "
              "than or equal to the 2nd dimension of Input(X)."
              "But received X's shape = [%s], Y's shape = [%s], "
              "and the second dimensions are %d and %d respectively.",
              x_dims,
              y_dims,
              x_dims[1],
              y_dims[1]));
    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ConvShiftGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ConvShiftGradOp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "ConvShiftGradOp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "ConvShiftGradOp");

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
  void Make() override {
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

$$Out[i] = \sum_{j=-(N-1)/2}^{(N-1)/2} X_{i+j} * Y_{j}$$

where X's index is computed modulo M, and Y's index is computed modulo N.

Both inputs X and Y can carry LoD (Level of Details) information.
However, the output only shares the LoD information with input X.

)DOC");
  }
};

template <typename T>
class ConvShiftKernel<platform::CPUPlace, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *X = context.Input<phi::DenseTensor>("X");
    auto *Y = context.Input<phi::DenseTensor>("Y");
    auto *Out = context.Output<phi::DenseTensor>("Out");
    Out->mutable_data<T>(context.GetPlace());

    auto x = EigenMatrix<T>::From(*X);
    auto y = EigenMatrix<T>::From(*Y);
    auto out = EigenMatrix<T>::From(*Out);
    out.setZero();

    size_t batch_size = X->dims()[0];
    size_t x_width = X->dims()[1];
    size_t y_width = Y->dims()[1];
    size_t y_half_width = (y_width - 1) / 2;

    for (size_t k = 0; k < batch_size; ++k) {
      for (size_t i = 0; i < x_width; ++i) {
        for (size_t j = 0; j < y_width; ++j) {
          int index = (i + j - y_half_width + x_width) % x_width;
          out(k, i) += x(k, index) * y(k, j);
        }
      }
    }
  }
};

template <typename T>
class ConvShiftGradKernel<platform::CPUPlace, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *X = context.Input<phi::DenseTensor>("X");
    auto *Y = context.Input<phi::DenseTensor>("Y");
    auto *dOut = context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto *dX = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *dY = context.Output<phi::DenseTensor>(framework::GradVarName("Y"));

    auto x = EigenMatrix<T>::From(*X);
    auto y = EigenMatrix<T>::From(*Y);
    auto dout = EigenMatrix<T>::From(*dOut);

    auto x_dims = X->dims();
    auto y_dims = Y->dims();
    size_t batch_size = x_dims[0];
    size_t x_width = x_dims[1];
    size_t y_width = y_dims[1];
    size_t y_half_width = (y_width - 1) / 2;

    // The below trades code duplication for efficiency (keeping the if
    // statement outside of the loop).
    if (dX) {
      dX->mutable_data<T>(context.GetPlace());
      auto dx = EigenMatrix<T>::From(*dX);
      dx.setZero();
      for (size_t k = 0; k < batch_size; ++k) {
        for (size_t i = 0; i < x_width; ++i) {
          for (size_t j = 0; j < y_width; ++j) {
            int index = (i + j - y_half_width + x_width) % x_width;
            dx(k, index) += dout(k, i) * y(k, j);
          }
        }
      }
    }

    if (dY) {
      dY->mutable_data<T>(context.GetPlace());
      auto dy = EigenMatrix<T>::From(*dY);
      dy.setZero();
      for (size_t k = 0; k < batch_size; ++k) {
        for (size_t i = 0; i < x_width; ++i) {
          for (size_t j = 0; j < y_width; ++j) {
            int index = (i + j - y_half_width + x_width) % x_width;
            dy(k, j) += x(k, index) * dout(k, i);
          }
        }
      }
    }
  }
};

template <typename T>
class ConvShiftGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("conv_shift_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(conv_shift,
                  ops::ConvShiftOp,
                  ops::ConvShiftOpMaker,
                  ops::ConvShiftGradOpMaker<paddle::framework::OpDesc>,
                  ops::ConvShiftGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(conv_shift_grad, ops::ConvShiftGradOp);
REGISTER_OP_CPU_KERNEL(conv_shift,
                       ops::ConvShiftKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    conv_shift_grad,
    ops::ConvShiftGradKernel<paddle::platform::CPUPlace, float>);
