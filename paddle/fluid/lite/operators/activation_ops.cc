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

#ifdef LITE_WITH_X86
#include "paddle/fluid/framework/operator.h"
#endif
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

class ActivationOp : public OpLite {
 public:
  explicit ActivationOp(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override { return true; }

  bool InferShape() const override {
    param_.Out->Resize(param_.X->dims());
    return true;
  }

  bool AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) override {
    auto X_name = opdesc.Input("X").front();
    auto Out_name = opdesc.Output("Out").front();

    param_.X = GetVar<lite::Tensor>(scope, X_name);
    param_.Out = GetMutableVar<Tensor>(scope, Out_name);
    return true;
  }

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "activation_op"; }

 private:
  mutable ActivationParam param_;
};

#ifdef LITE_WITH_X86
class ActivationGradOp : public OpLite {
 public:
  explicit ActivationGradOp(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override {
    CHECK_OR_FALSE(param_.X_grad);
    CHECK_OR_FALSE(param_.Out_grad);
    return true;
  }

  bool InferShape() const override {
    param_.X_grad->Resize(param_.Out_grad->dims());
    return true;
  }

  bool AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) override {
    auto Out_grad_name = opdesc.Input(framework::GradVarName("Out")).front();
    auto X_grad_name = opdesc.Output(framework::GradVarName("X")).front();

    param_.Out_grad = GetVar<lite::Tensor>(scope, Out_grad_name);
    param_.X_grad = GetMutableVar<Tensor>(scope, X_grad_name);

    if (opdesc.HasInput("X")) {
      auto X_name = opdesc.Input("X").front();
      param_.X = GetVar<lite::Tensor>(scope, X_name);
    } else {
      param_.X = param_.X_grad;
    }

    if (opdesc.HasInput("Out")) {
      auto Out_name = opdesc.Input("Out").front();
      param_.Out = GetVar<lite::Tensor>(scope, Out_name);
    } else {
      param_.Out = param_.Out_grad;
    }

    return true;
  }

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "activation_grad_op"; }

 private:
  mutable ActivationGradParam param_;
};
#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(square, paddle::lite::operators::ActivationOp);
#ifdef LITE_WITH_X86
REGISTER_LITE_OP(square_grad, paddle::lite::operators::ActivationGradOp);
#endif
