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

class NormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) A tensor of rank >= axis.");
    AddAttr<int>("axis",
                 "The axis on which to apply normalization. If axis < 0, "
                 "the dimension to normalization is rank(X) + axis. -1 is "
                 "the last dimension.");
    AddAttr<float>("epsilon",
                   "(float, default 1e-10) The epsilon value is used "
                   "to avoid division by zero.")
        .SetDefault(1.0e-10f);
    AddOutput("Norm",
              "(Tensor) A tensor saved the `sqrt(sum(x) + epsion)` will "
              "be used in backward kernel.")
        .AsIntermediate();
    AddOutput("Out", "(Tensor) A tensor of the same shape as X.");
    AddComment(R"DOC(

Given a tensor, apply 2-normalization along the provided axis.

$$
y = \frac{x}{ \sqrt{\sum {x^2} + epsion }}
$$

where, $\sum {x^2}$ is calculated along the `axis` dimension.
        
)DOC");
  }
};

class NormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of NormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of NormOp should not be null.");
    auto xdim = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", xdim);
    int axis = ctx->Attrs().Get<int>("axis");
    if (axis < 0) axis = xdim.size() + axis;
    xdim[axis] = 1;
    ctx->SetOutputDim("Norm", xdim);
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
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(norm, ops::NormOp, ops::NormOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(norm_grad, ops::NormOpGrad);
REGISTER_OP_CPU_KERNEL(norm, ops::NormKernel<CPU, float>,
                       ops::NormKernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(norm_grad, ops::NormGradKernel<CPU, float>,
                       ops::NormGradKernel<CPU, double>);
