/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/gumbel_softmax_op.h"
#include <string>
#include <unordered_map>
#include "paddle/fluid/operators/common_infer_shape_functions.h"

namespace paddle {
namespace operators {
class GumbelSoftmaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    return UnaryOpUnchangedInferShapeCheckAxis(ctx);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class GumbelSoftmaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) An N-D Tensor, N >= 1,"
             "The first N - 1 dimensions index into a batch of independent "
             "distributions "
             "and the last dimension represents a vector of probabilities for "
             "each class.");
    AddOutput("Out", "The sampled tensor with the same shape as X.");
    AddAttr<float>("temperature",
                   "(float, default 1.0) non-negative scalar temperature.")
        .SetDefault(1.0);
    AddAttr<bool>(
        "hard",
        "(bool, default false) "
        "if True, the returned samples will be discretized as one-hot vectors, "
        "but will be differentiated as if it is the soft sample in autograd.")
        .SetDefault(false);
    AddAttr<int>("axis",
                 "(int, default -1)"
                 "The dimension index of Input(x) to perform gumbel_softmax.")
        .SetDefault(-1);
    AddComment(R"DOC(
GumbelSoftmax Operator.

Samples from the Gumbel-Softmax distribution and optionally discretizes.

)DOC");
  }
};

class GumbelSoftmaxGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "gumbel_softmax_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "gumbel_softmax_grad");
    PADDLE_ENFORCE_EQ(
        ctx->GetInputDim("Out"),
        ctx->GetInputDim(framework::GradVarName("Out")),
        platform::errors::InvalidArgument("Input(Out) and its gradients "
                                          "should have the same shape."));

    ctx->SetOutputDim(framework::GradVarName("X"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
  }
};

template <typename T>
class GumbelSoftmaxGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("gumbel_softmax_grad");
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(gumbel_softmax, ops::GumbelSoftmaxOp,
                  ops::GumbelSoftmaxOpMaker,
                  ops::GumbelSoftmaxGradOpMaker<paddle::framework::OpDesc>,
                  ops::GumbelSoftmaxGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(gumbel_softmax_grad, ops::GumbelSoftmaxGradOp);

REGISTER_OP_CPU_KERNEL(
    gumbel_softmax,
    ops::GumbelSoftmaxKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GumbelSoftmaxKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    gumbel_softmax_grad,
    ops::GumbelSoftmaxGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GumbelSoftmaxGradKernel<paddle::platform::CPUDeviceContext, double>);
