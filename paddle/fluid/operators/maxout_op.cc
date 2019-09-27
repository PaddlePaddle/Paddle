/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/operators/maxout_op.h"
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class MaxOutOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor of maxout operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddOutput("Out",
              "(Tensor) The output tensor of maxout operator."
              "The format of output tensor is also NCHW."
              "Where N is batch size, C is "
              "the number of channels, H and W is the height and "
              "width of feature.");
    AddAttr<int>(
        "groups",
        "(int),"
        "Specifies how many groups the input tensor will be split"
        "in the channel dimension. And the number of output channel is "
        "the number of channels divided by groups.");
    AddComment(R"DOC(
MaxOut Operator.

Assumed the input shape is (N, Ci, H, W).
The output shape is (N, Co, H, W).
Then $Co = Ci / groups$ and the operator formula is as follows:

$$ y_{si+j} = \max_{k} x_{gsi + sk + j} $$
$$ g = groups $$
$$ s = \\frac{input.size}{num\\_channels} $$
$$ 0 \\le i < \\frac{num\\_channels}{groups} $$
$$ 0 \\le j < s $$
$$ 0 \\le k < groups $$

Please refer to Paper:
  - Maxout Networks: http://www.jmlr.org/proceedings/papers/v28/goodfellow13.pdf
  - Multi-digit Number Recognition from Street View \
    Imagery using Deep Convolutional Neural Networks: \
    https://arxiv.org/pdf/1312.6082v4.pdf

)DOC");
  }
};

class MaxOutOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of MaxoutOpshould not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MaxoutOp should not be null.");
    auto in_x_dims = ctx->GetInputDim("X");
    int groups = ctx->Attrs().Get<int>("groups");
    // check groups > 1
    PADDLE_ENFORCE_GT(groups, 1, "groups should be larger than 1 in maxoutop");
    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1] / groups});
    output_shape.push_back(in_x_dims[2]);
    output_shape.push_back(in_x_dims[3]);
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }
};

class MaxOutOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of MaxOutOpGrad must not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(Grad@X) of MaxOutOpGrad should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(maxout, ops::MaxOutOp, ops::MaxOutOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(maxout_grad, ops::MaxOutOpGrad);
REGISTER_OP_CPU_KERNEL(
    maxout, ops::MaxOutKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    maxout_grad,
    ops::MaxOutGradKernel<paddle::platform::CPUDeviceContext, float>);
