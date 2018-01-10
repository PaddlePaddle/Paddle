/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/cross_channel_norm_op.h"
namespace paddle {
namespace operators {

template <typename AttrType>
class CrossChannelNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CrossChannelNormOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "(Tensor) The input tensor of cross_channel_norm operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddAttr<AttrType>("scale", "(float), increase or reduce");
    AddAttr<AttrType>("epsilon",
                      "(float, default 1e-10) Constant "
                      "for numerical stability.")
        .SetDefault(1.0e-10f);
    AddOutput("Out",
              "(Tensor) The output tensor of cross_channel_norm operator."
              "N * M."
              "M = C * H * W");
    AddComment(R"DOC(
       "Input shape: $(N, C, H, W)$
        Sclae shape: $(C, 1)$
        Output shape: $(N, C, H, W)$
        Where
        $$
        forward \\
         y_{1} = \frac {x_{1}}{\sqrt{\sum{x_{i}^{2}}}} \\
         y_{2} = \frac {x_{2}}{\sqrt{\sum{x_{i}^{2}}}} \\
         ... \\
         y_{n} = \frac {x_{n}}{\sqrt{\sum{x_{i}^{2}}}} \\
        backward \\
            \frac{\frac{\mathrm{d}L }{\mathrm{d}y_{1}} - \frac {x_{1}\sum {\frac{\mathrm{d} L}{\mathrm{d} y_{j}}}x_{j}}{\sum x_{j}^{2}} }{\sqrt{\sum{x_{j}^{2}}}}
        $$
        )DOC");
  }
};

class CrossChannelNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of CrossChannelNormOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of CrossChannelNormOp should not be null.");
    auto in_x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", in_x_dims);
  }
};

class CrossChannelNormOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Input(X@GRAD) should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(cross_channel_norm, ops::CrossChannelNormOp,
            ops::CrossChannelNormOpMaker<float>, cross_channel_norm_grad,
            ops::CrossChannelNormOpGrad);
REGISTER_OP_CPU_KERNEL(
    cross_channel_norm,
    ops::CrossChannelNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CrossChannelNormKernel<paddle::platform::CPUDeviceContext, double,
                                float>);
REGISTER_OP_CPU_KERNEL(
    cross_channel_norm_grad,
    ops::CrossChannelNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CrossChannelNormGradKernel<paddle::platform::CPUDeviceContext, double,
                                    float>);
