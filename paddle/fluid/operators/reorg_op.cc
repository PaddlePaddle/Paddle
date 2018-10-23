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

#include "paddle/fluid/operators/reorg_op.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class ReorgOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of reorgOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of reorgOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 4, "input should be a 4D tensor");
    auto stride = ctx->Attrs().Get<int64_t>("stride");

    PADDLE_ENFORCE_GT(stride, 0, "The stride should be Greater than 0");
    PADDLE_ENFORCE_GT(x_dims[1], 0, "input channel should be Greater than 0");
    PADDLE_ENFORCE_GT(x_dims[2], 0, "input Height should be Greater than 0");
    PADDLE_ENFORCE_GT(x_dims[3], 0, "input Width should be Greater than 0");

    PADDLE_ENFORCE_EQ(
        x_dims[1] % (stride * stride), 0,
        "input channel should be dvisible of the square of reorg stride");
    PADDLE_ENFORCE_EQ(
        x_dims[2] % (stride), 0,
        "input Height should be dvisible of the square of reorg stride");
    PADDLE_ENFORCE_EQ(
        x_dims[3] % (stride), 0,
        "input Width should be dvisible of the square of reorg stride");

    VLOG(3) << "reorg operator x.shape=" << x_dims << "Attribute stride"
            << stride << std::endl;

    std::vector<int64_t> output_shape(4, 0);  // [B,C,H,W]
    output_shape[0] = x_dims[0];
    output_shape[1] = x_dims[1] * stride * stride;
    output_shape[2] = x_dims[2] / stride;
    output_shape[3] = x_dims[3] / stride;

    auto out_dims = framework::make_ddim(output_shape);

    ctx->SetOutputDim("Out", out_dims);

    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }
};

class ReorgOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor). The input should be a 4D tensor B * C * W * H of reorg "
             "operator.");
    AddOutput("Out",
              "(Tensor), The output should be a 4D tensor B * C2 * W2 * H2 of "
              "reorg operator.");
    AddAttr<int64_t>("stride",
                     "(int64_t, default 1) stride used to do reorgnization.")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddComment(R"DOC(
        reorg operator used in Yolo v2.
        The equation is: C2 = C1/stride * stride, W2 = W1 ∗ stride + offset % stride, H2 = H1 ∗ stride + offset / stride, 

        Reshape Input(X) into the shape according to Attr(stride). The
        data in Input(X) are unchanged.

        Examples:

            1. Given a 4-D tensor Input(X) with a shape [128, 2048, 26, 26], and the stride is 2, the reorg operator will transform Input(X)
            into a 4-D tensor with shape [128, 2048, 13, 13] and leaving Input(X)'s data unchanged.

    )DOC");
  }
};

class ReorgGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(reorg, ops::ReorgOp, ops::ReorgOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(reorg_grad, ops::ReorgGradOp);
REGISTER_OP_CPU_KERNEL(
    reorg, ops::ReorgKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ReorgKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ReorgKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    reorg_grad, ops::ReorgGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ReorgGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ReorgGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
