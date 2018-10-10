/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/pow_op.h"
#include <string>
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

class PowOp : public framework::OperatorWithKernel {
 public:
  PowOp(const std::string &type, const framework::VariableNameMap &inputs,
        const framework::VariableNameMap &outputs,
        const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of PowOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of PowOp should not be null.");
    ctx->ShareDim("X", /*->*/ "Out");
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class PowOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of Pow operator.");
    AddOutput("Out", "(Tensor) Output tensor of Pow operator.");
    AddAttr<float>("factor", "The exponential factor of Pow").SetDefault(1.0f);
    AddComment(R"DOC(
**Pow operator**

$out = x^{factor}$

)DOC");
  }
};

class PowOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    auto &in_var_name = op_desc.Input("X").front();
    auto out_var_name = op_desc.Output("Out").front();

    auto &in_var = detail::Ref(block->FindVarRecursive(in_var_name));
    auto *out_var = block->FindVarRecursive(out_var_name);

    if (in_var_name != out_var_name) {
      out_var->SetType(in_var.GetType());
      out_var->SetDataType(in_var.GetDataType());
    }
  }
};

class PowOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->ShareDim("X", framework::GradVarName("X"));
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }
};

class PowGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("pow_grad");
    grad_op->SetInput("X", Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    grad_op->SetAttrMap(Attrs());
    grad_op->SetOutput(framework::GradVarName("X"), InputGrad("X"));

    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

template <typename T>
struct PowFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext &context,
                  const framework::Tensor &x, float factor,
                  framework::Tensor *out) const {
    const T *src_ptr = x.data<T>();
    T *dst_ptr = out->data<T>();
    int64_t numel = x.numel();
    PADDLE_ENFORCE_EQ(numel, out->numel());

    for (int64_t i = 0; i < numel; ++i) {
      dst_ptr[i] = std::pow(src_ptr[i], factor);
    }
  }
};

template <typename T>
struct PowGradFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext &context,
                  const framework::Tensor &x, const framework::Tensor &d_out,
                  float factor, framework::Tensor *d_x) const {
    const T *x_ptr = x.data<T>();
    const T *d_out_ptr = d_out.data<T>();
    T *d_x_ptr = d_x->data<T>();
    int64_t numel = x.numel();
    PADDLE_ENFORCE_EQ(numel, d_out.numel());
    PADDLE_ENFORCE_EQ(numel, d_x->numel());

    for (int64_t i = 0; i < numel; ++i) {
      d_x_ptr[i] = d_out_ptr[i] * factor * std::pow(x_ptr[i], factor - 1);
    }
  }
};

template struct PowFunctor<platform::CPUDeviceContext, float>;
template struct PowFunctor<platform::CPUDeviceContext, double>;

template struct PowGradFunctor<platform::CPUDeviceContext, float>;
template struct PowGradFunctor<platform::CPUDeviceContext, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(pow, ops::PowOp, ops::PowOpMaker, ops::PowGradMaker,
                  ops::PowOpVarTypeInference);
REGISTER_OPERATOR(pow_grad, ops::PowOpGrad);

REGISTER_OP_CPU_KERNEL(pow, ops::PowKernel<plat::CPUDeviceContext, float>,
                       ops::PowKernel<plat::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(pow_grad,
                       ops::PowGradKernel<plat::CPUDeviceContext, float>,
                       ops::PowGradKernel<plat::CPUDeviceContext, double>);
