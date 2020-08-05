/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/log_softmax_op.h"

namespace paddle {
namespace operators {

class LogSoftmaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X", "log_softmax");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "log_softmax");

    auto dim_x = ctx->GetInputDim("X");
    auto rank_x = dim_x.size();
    auto axis = ctx->Attrs().Get<int>("axis");
    PADDLE_ENFORCE_GE(
        axis, -rank_x,
        platform::errors::InvalidArgument(
            "Attr(axis) value should be in range [-R, R-1], "
            "R is the rank of Input(X). But received axis: %d, R: %d.",
            axis, rank_x));
    PADDLE_ENFORCE_LT(
        axis, rank_x,
        platform::errors::InvalidArgument(
            "Attr(axis) value should be in range [-R, R-1], "
            "R is the rank of Input(X). But received axis: %d, R: %d.",
            axis, rank_x));

    ctx->SetOutputDim("Out", dim_x);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class LogSoftmaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of softmax, "
             "whose dimension :attr:`axis` is the input_feature_dimensions.");
    AddOutput("Out", "The normalized values with the same shape as X.");
    AddAttr<int>("axis",
                 "The dimension index of Input(x) to perform log_softmax,"
                 "default -1 for last dimension")
        .SetDefault(-1);
    AddComment(R"DOC(
LogSoftmax Operator.

)DOC");
  }
};

class LogSoftmaxOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "log_softmax_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@grad", "log_softmax_grad");
    PADDLE_ENFORCE_EQ(
        ctx->GetInputDim("Out"),
        ctx->GetInputDim(framework::GradVarName("Out")),
        platform::errors::InvalidArgument("Input(Out) and its gradients "
                                          "should have the same shape."));

    ctx->SetOutputDim(framework::GradVarName("X"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class LogSoftmaxOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("log_softmax_grad");
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(SoftmaxInplaceInferer, {"X", "Out"});

// NOTE(zjl): AVX implementation of SoftmaxGrad does not support in-place
DECLARE_CUDA_ONLY_INPLACE_OP_INFERER(SoftmaxGradInplaceInferer,
                                     {"Out", framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(log_softmax, ops::LogSoftmaxOp, ops::LogSoftmaxOpMaker,
                  ops::LogSoftmaxOpInferVarType,
                  ops::LogSoftmaxOpGradMaker<paddle::framework::OpDesc>,
                  ops::LogSoftmaxOpGradMaker<paddle::imperative::OpBase>,
                  ops::LogSoftmaxInplaceInferer);
REGISTER_OPERATOR(log_softmax_grad, ops::LogSoftmaxOpGrad,
                  ops::SoftmaxGradInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    log_softmax,
    ops::LogSoftmaxKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LogSoftmaxKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    log_softmax_grad,
    ops::LogSoftmaxGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LogSoftmaxGradKernel<paddle::platform::CPUDeviceContext, double>);
