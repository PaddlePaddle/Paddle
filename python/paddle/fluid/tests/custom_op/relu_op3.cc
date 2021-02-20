// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class Relu3Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Y", in_dims);
  }
};

class Relu3OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddOutput("Y", "Output of relu_op");
    AddComment(R"DOC(
Relu3 Operator.
)DOC");
  }
};

class Relu3GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
  }
};

template <typename T>
class Relu3GradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("relu3_grad");
    op->SetInput("Y", this->Output("Y"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class Relu3Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<Tensor>("X");
    auto* out_t = ctx.Output<Tensor>("Y");
    auto x = in_t->data<T>();
    auto y = out_t->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < in_t->numel(); ++i) {
      y[i] = std::max(static_cast<T>(0.), x[i]);
    }
  }
};

template <typename DeviceContext, typename T>
class Relu3GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dy_t = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* y_t = ctx.Input<Tensor>("Y");
    auto* dx_t = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto dy = dy_t->data<T>();
    auto y = y_t->data<T>();
    auto dx = dx_t->mutable_data<T>(ctx.GetPlace());

    for (int i = 0; i < y_t->numel(); ++i) {
      dx[i] = dy[i] * (y[i] > static_cast<T>(0) ? 1. : 0.);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
REGISTER_OPERATOR(relu3,
                  ops::Relu3Op,
                  ops::Relu3OpMaker,
                  ops::Relu3GradMaker<paddle::framework::OpDesc>,
                  ops::Relu3GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(relu3_grad, ops::Relu3GradOp);
REGISTER_OP_CPU_KERNEL(relu3,
                       ops::Relu3Kernel<CPU, float>,
                       ops::Relu3Kernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(relu3_grad,
                       ops::Relu3GradKernel<CPU, float>,
                       ops::Relu3GradKernel<CPU, double>);
