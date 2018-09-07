/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/norm_op.h"
namespace paddle {
namespace operators {

template <typename AttrType>
class NormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor of norm operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddInput("Scale",
             "(Tensor) The input tensor of norm operator. "
             "The format of input tensor is C * 1.");
    AddAttr<AttrType>("epsilon",
                      "(float, default 1e-10) Constant "
                      "for numerical stability.")
        .SetDefault(1.0e-10f);
    AddOutput("Out",
              "(Tensor) The output tensor of norm operator."
              "N * M."
              "M = C * H * W");
    AddComment(R"DOC(
       "Input shape: $(N, C, H, W)$
        Scale shape: $(C, 1)$
        Output shape: $(N, C, H, W)$
        Where
        forward
          $$
            [\frac {x_{1}}{\sqrt{\sum{x_{i}^{2}}}} \frac {x_{2}}{\sqrt{\sum{x_{i}^{2}}}} \frac {x_{3}}{\sqrt{\sum{x_{i}^{2}}}} \cdot  \cdot  \cdot \frac {x_{n}}{\sqrt{\sum{x_{i}^{2}}}}]
          $$
        backward
          $$
            \frac{\frac{\mathrm{d}L }{\mathrm{d}y_{1}} - \frac {x_{1}\sum {\frac{\mathrm{d} L}{\mathrm{d} y_{j}}}x_{j}}{\sum x_{j}^{2}} }{\sqrt{\sum{x_{j}^{2}}}}
          $$
        )DOC");
  }
};

class NormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of NormOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Scale"),
                   "Input(Scale) of NormOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of NormOp should not be null.");
    auto in_x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", in_x_dims);
  }
};

class NormOpGrad : public framework::OperatorWithKernel {
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
REGISTER_OPERATOR(norm, ops::NormOp, ops::NormOpMaker<float>,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(norm_grad, ops::NormOpGrad);
REGISTER_OP_CPU_KERNEL(
    norm, ops::NormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::NormKernel<paddle::platform::CPUDeviceContext, double, float>);
REGISTER_OP_CPU_KERNEL(
    norm_grad, ops::NormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::NormGradKernel<paddle::platform::CPUDeviceContext, double, float>);
