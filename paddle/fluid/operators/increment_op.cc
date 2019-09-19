//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/increment_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class IncrementOp : public framework::OperatorWithKernel {
 public:
  IncrementOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of IncrementOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of IncrementOp should not be null.");
    PADDLE_ENFORCE_EQ(1, framework::product(ctx->GetInputDim("X")));
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    // IncrementOp kernel's device type is decided by input tensor place
    kt.place_ = ctx.Input<framework::LoDTensor>("X")->place();
    return kt;
  }
};

class IncrementOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensor of increment operator");
    AddOutput("Out", "(Tensor) The output tensor of increment operator.");
    AddAttr<float>("step",
                   "(float, default 1.0) "
                   "The step size by which the "
                   "input tensor will be incremented.")
        .SetDefault(1.0);
    AddComment(R"DOC(
Increment Operator.

The equation is: 
$$Out = X + step$$

)DOC");
  }
};

template <typename T>
class IncrementGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("increment");
    grad_op->SetInput("X", this->Output("Out"));
    grad_op->SetOutput("Out", this->Input("X"));
    grad_op->SetAttr("step", -boost::get<float>(this->GetAttr("step")));
    return std::unique_ptr<T>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(increment, ops::IncrementOp, ops::IncrementOpMaker,
                  ops::IncrementGradOpMaker<paddle::framework::OpDesc>,
                  ops::IncrementGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    increment, ops::IncrementKernel<paddle::platform::CPUDeviceContext, float>,
    ops::IncrementKernel<paddle::platform::CPUDeviceContext, double>,
    ops::IncrementKernel<paddle::platform::CPUDeviceContext, int>,
    ops::IncrementKernel<paddle::platform::CPUDeviceContext, int64_t>);
