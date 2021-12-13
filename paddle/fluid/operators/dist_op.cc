// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/dist_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace operators {

class DistOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Dist");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "Dist");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Dist");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_NE(framework::product(x_dims), 0,
                      platform::errors::InvalidArgument(
                          "The Input(X) has not been initialized properly. The "
                          "shape of Input(X) = [%s].",
                          x_dims));
    PADDLE_ENFORCE_NE(framework::product(y_dims), 0,
                      platform::errors::InvalidArgument(
                          "The Input(Y) has not been initialized properly. The "
                          "shape of Input(Y) = [%s].",
                          y_dims));
    ctx->SetOutputDim("Out", {1});
  }
};

class DistOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input Tensor of Dist Op.");
    AddInput("Y", "The Right-hand-side input Tensor of Dist Op.");
    AddOutput("Out",
              "The output of Dist Op, "
              "which is the p-norm of (X - Y)");
    AddAttr<float>("p", "the norm to be computed.").SetDefault(2.0f);
    AddComment(R"DOC(
Dist Operator.
Given two tensors X and Y, compute Lp-norm of (X-Y). It is not a norm in a strict sense,
only as a measure of distance. The shapes of X and Y must be broadcastable. Where, Z = X - Y,

When p = 0, defining $0^0 = 0$, the zero-norm of Z is simply the number of non-zero elements of z.
$$
||Z||_{0} = \lim_{p \rightarrow 0} \sum_{i=1}^{m} |z_i|^p
$$

When p = inf, the inf-norm of Z is the maximum element of Z.
$$
||Z||_\infty=\max_i |z_i|
$$

When p = -inf, the negative-inf-norm of Z is the minimum element of Z.
$$
||Z||_{-\infty}=\min_i |z_i|
$$

Otherwise, the p-norm of Z follows the formula,
$$
||Z||_{p} = (\sum_{i=i}^{m} |z_i|^p)^{1/p}
$$
    )DOC");
  }
};

class DistOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Y"))) {
      ctx->SetOutputDim(framework::GradVarName("Y"), y_dims);
    }
  }
};

template <typename T>
class DistGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(dist, ops::DistOp, ops::DistOpMaker,
                  ops::DistGradOpMaker<paddle::framework::OpDesc>,
                  ops::DistGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(dist_grad, ops::DistOpGrad);
REGISTER_OP_CPU_KERNEL(
    dist, ops::DistKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DistKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    dist_grad, ops::DistGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DistGradKernel<paddle::platform::CPUDeviceContext, double>)
