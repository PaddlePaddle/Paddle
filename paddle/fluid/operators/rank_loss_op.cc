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

#include "paddle/fluid/operators/rank_loss_op.h"

#include <string>

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

class RankLossOp : public framework::OperatorWithKernel {
 public:
  RankLossOp(const std::string &type, const framework::VariableNameMap &inputs,
             const framework::VariableNameMap &outputs,
             const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "RankLoss");
    OP_INOUT_CHECK(ctx->HasInput("Left"), "Input", "Left", "RankLoss");
    OP_INOUT_CHECK(ctx->HasInput("Right"), "Input", "Right", "RankLoss");

    auto label_dims = ctx->GetInputDim("Label");
    auto left_dims = ctx->GetInputDim("Left");
    auto right_dims = ctx->GetInputDim("Right");
    // check label_dims valid
    PADDLE_ENFORCE_GE(
        label_dims.size(), 1,
        platform::errors::InvalidArgument(
            "The dimension size of Input(Label) must be greater than "
            "or equal to 1, but received %d.",
            label_dims.size()));
    PADDLE_ENFORCE_LE(
        label_dims.size(), 2,
        platform::errors::InvalidArgument("The dimension size of Input(Label) "
                                          "must be less than or equal to 2, "
                                          "but received %d.",
                                          label_dims.size()));
    if (label_dims.size() == 2U) {
      PADDLE_ENFORCE_EQ(
          label_dims[1], 1,
          platform::errors::InvalidArgument(
              "The last dimension of Input(Label) must be 1, but received %d.",
              label_dims[1]));
    }
    // check left_dims valid
    PADDLE_ENFORCE_GE(
        left_dims.size(), 1,
        platform::errors::InvalidArgument(
            "The dimension size of Input(Left) must be greater than "
            "or equal to 1, but received %d.",
            left_dims.size()));
    PADDLE_ENFORCE_LE(
        left_dims.size(), 2,
        platform::errors::InvalidArgument("The dimension size of Input(Left) "
                                          "must be less than or equal to 2, "
                                          "but received %d.",
                                          left_dims.size()));
    if (left_dims.size() == 2U) {
      PADDLE_ENFORCE_EQ(
          left_dims[1], 1,
          platform::errors::InvalidArgument(
              "The last dimension of Input(Left) must be 1, but received %d.",
              left_dims[1]));
    }
    // check right_dims valid
    PADDLE_ENFORCE_GE(
        right_dims.size(), 1,
        platform::errors::InvalidArgument(
            "The dimension size of Input(Right) must be greater than "
            "or equal to 1, but received %d.",
            right_dims.size()));
    PADDLE_ENFORCE_LE(
        right_dims.size(), 2,
        platform::errors::InvalidArgument("The dimension size of Input(Right) "
                                          "must be less than or equal to 2, "
                                          "but received %d.",
                                          right_dims.size()));
    if (right_dims.size() == 2U) {
      PADDLE_ENFORCE_EQ(
          right_dims[1], 1,
          platform::errors::InvalidArgument(
              "The last dimension of Input(Right) must be 1, but received %d.",
              right_dims[1]));
    }
    PADDLE_ENFORCE_EQ(
        label_dims[0], left_dims[0],
        platform::errors::InvalidArgument(
            "The first dimension of Input(Label) and Input(Left) "
            "must have the same value. But received Label.dims[0]=%d, "
            "Left.dims[0]=%d.",
            label_dims[0], left_dims[0]));
    PADDLE_ENFORCE_EQ(
        label_dims[0], right_dims[0],
        platform::errors::InvalidArgument(
            "The first dimension of Input(Label) and Input(Right) "
            "must have the same value. But received Label.dims[0]=%d, "
            "Right.dims[0]=%d.",
            label_dims[0], right_dims[0]));
    ctx->SetOutputDim("Out", label_dims);
  }
};

class RankLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Label",
             "(2-D Tensor with shape [batch_size x 1]) "
             "The label indicating A ranked higher than B or not.");
    AddInput("Left",
             "(2-D Tensor with shape [batch_size x 1]) "
             "The output of RankNet for doc A.");
    AddInput("Right",
             "(2-D Tensor with shape [batch_size x 1]) "
             "The output of RankNet for doc B.");
    AddOutput("Out",
              "(2-D Tensor with shape [batch_size x 1]) "
              "The output loss of RankLoss operator.");
    AddComment(R"DOC(
RankLoss Operator.

RankLoss operator for RankNet
(http://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf). 
RankNet is a pairwise ranking model with
one training sample consisting of a pair of doc A and B, and the label P
indicating that A is ranked higher than B or not:

P = {0, 1} or {0, 0.5, 1}, where 0.5 means no information about the rank of
the input pair.

The RankLoss operator takes three inputs: Left (o_i), Right (o_j) and Label
(P_{i,j}), which represent the output score of RankNet for the two docs and 
the label respectively, and yields the rank loss C_{i,j} using the following 
equation:

$$
  C_{i,j} = -\tilde{P_{ij}} * o_{i,j} + \log(1 + e^{o_{i,j}}) \\
  o_{i,j} =  o_i - o_j  \\
  \tilde{P_{i,j}} = \left \{0, 0.5, 1 \right \} \ or \ \left \{0, 1 \right \}
$$

The operator can take batch inputs with size batch_size (batch_size >= 1).

)DOC");
  }
};

class RankLossGradOp : public framework::OperatorWithKernel {
 public:
  RankLossGradOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "RankLossGrad");
    OP_INOUT_CHECK(ctx->HasInput("Left"), "Input", "Left", "RankLossGrad");
    OP_INOUT_CHECK(ctx->HasInput("Right"), "Input", "Right", "RankLossGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "RankLossGrad");

    auto left_dims = ctx->GetInputDim("Left");
    auto right_dims = ctx->GetInputDim("Right");
    auto left_grad_name = framework::GradVarName("Left");
    auto right_grad_name = framework::GradVarName("Right");

    if (ctx->HasOutput(left_grad_name)) {
      ctx->SetOutputDim(left_grad_name, left_dims);
    }

    if (ctx->HasOutput(right_grad_name)) {
      ctx->SetOutputDim(right_grad_name, right_dims);
    }
  }
};

template <typename T>
class RankLossGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("rank_loss_grad");
    op->SetInput("Label", this->Input("Label"));
    op->SetInput("Left", this->Input("Left"));
    op->SetInput("Right", this->Input("Right"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("Left"), this->InputGrad("Left"));
    op->SetOutput(framework::GradVarName("Right"), this->InputGrad("Right"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(rank_loss, ops::RankLossOp, ops::RankLossOpMaker,
                  ops::RankLossGradMaker<paddle::framework::OpDesc>,
                  ops::RankLossGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(rank_loss_grad, ops::RankLossGradOp);
REGISTER_OP_CPU_KERNEL(
    rank_loss, ops::RankLossKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    rank_loss_grad,
    ops::RankLossGradKernel<paddle::platform::CPUDeviceContext, float>);

REGISTER_OP_CUDA_KERNEL(rank_loss,
                        paddle::operators::RankLossKernel<
                            paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(rank_loss_grad,
                        paddle::operators::RankLossGradKernel<
                            paddle::platform::CUDADeviceContext, float>);
