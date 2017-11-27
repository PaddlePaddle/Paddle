/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

class IncrementInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of IncrementOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of IncrementOp should not be null.");
    PADDLE_ENFORCE_EQ(1, framework::product(ctx->GetInputDim("X")));
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }
};

struct IncrementFunctor {
  IncrementFunctor(const framework::LoDTensor &x, framework::LoDTensor *out,
                   float value)
      : x_(x), out_(out), value_(value) {}

  template <typename T>
  void operator()() const {
    *out_->data<T>() = *x_.data<T>() + static_cast<T>(value_);
  }

  const framework::LoDTensor &x_;
  framework::LoDTensor *out_;
  float value_;
};

class IncrementOp : public framework::OperatorBase {
 public:
  IncrementOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto &x = scope.FindVar(Input("X"))->Get<framework::LoDTensor>();
    auto &out =
        *scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();

    PADDLE_ENFORCE(platform::is_cpu_place(x.place()));
    out.Resize(x.dims());
    out.mutable_data(x.place(), x.type());
    float value = Attr<float>("step");
    framework::VisitDataType(framework::ToDataType(out.type()),
                             IncrementFunctor(x, &out, value));
  }
};

class IncrementOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  IncrementOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
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

class IncrementGradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDescBind> Apply() const override {
    auto *grad_op = new framework::OpDescBind();
    grad_op->SetType("increment");
    grad_op->SetInput("X", Output("Out"));
    grad_op->SetOutput("Out", Input("X"));
    grad_op->SetAttr("step", -boost::get<float>(GetAttr("step")));
    return std::unique_ptr<framework::OpDescBind>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(increment, ops::IncrementOp, ops::IncrementInferShape,
                  ops::IncrementOpMaker, ops::IncrementGradOpMaker);
