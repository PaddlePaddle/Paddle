// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

class Relu2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Y", in_dims);
  }
};

class Relu2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddOutput("Y", "Output of relu_op");
    AddComment(R"DOC(
Relu2 Operator.
)DOC");
  }
};

class Relu2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    ctx->SetOutputDim(framework::GradVarName("X"), in_dims);
  }
};

template <typename T>
class Relu2GradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  std::unique_ptr<T> Apply() const override {
    auto* op = new T();
    op->SetType("relu2_grad");
    op->SetInput("Y", this->Output("Y"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    return std::unique_ptr<T>(op);
  }
};

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class Relu2Kernel : public framework::OpKernel<T> {
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
class Relu2GradKernel : public framework::OpKernel<T> {
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
REGISTER_OPERATOR(relu2,
                  ops::Relu2Op,
                  ops::Relu2OpMaker,
                  ops::Relu2GradMaker<paddle::framework::OpDesc>,
                  ops::Relu2GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(relu2_grad, ops::Relu2GradOp);
REGISTER_OP_CPU_KERNEL(relu2,
                       ops::Relu2Kernel<CPU, float>,
                       ops::Relu2Kernel<CPU, double>);
REGISTER_OP_CPU_KERNEL(relu2_grad,
                       ops::Relu2GradKernel<CPU, float>,
                       ops::Relu2GradKernel<CPU, double>);
