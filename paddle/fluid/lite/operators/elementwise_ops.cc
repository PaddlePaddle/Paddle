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

#include "paddle/fluid/lite/operators/elementwise_ops.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ElementwiseOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool ElementwiseOp::InferShape() const {
  CHECK_OR_FALSE(param_.X->dims().size() >= param_.Y->dims().size());
  param_.Out->Resize(param_.X->dims());
  param_.Out->raw_tensor().set_lod(param_.X->lod());
  return true;
}

bool ElementwiseOp::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Y_name = opdesc.Input("Y").front();
  auto Out_name = opdesc.Output("Out").front();

  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Y = GetVar<lite::Tensor>(scope, Y_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);
  param_.axis = opdesc.GetAttr<int>("axis");
  return true;
}

#ifdef LITE_WITH_X86
bool ElementwiseGradExplicitOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.X_grad);
  CHECK_OR_FALSE(param_.Out_grad);
  return true;
}

bool ElementwiseGradExplicitOp::InferShape() const {
  param_.X_grad->Resize(param_.Out_grad->dims());
  if (param_.Y_grad) param_.Y_grad->Resize(param_.Y->dims());
  return true;
}

bool ElementwiseGradExplicitOp::AttachImpl(const cpp::OpDesc& opdesc,
                                           lite::Scope* scope) {
  CHECK_EQ(opdesc.InputArgumentNames().size(), 2UL);
  auto Y_name = opdesc.Input("Y").front();
  auto Out_name = opdesc.Input(framework::GradVarName("Out")).front();
  auto X_grad = opdesc.Output(framework::GradVarName("X")).front();

  if (opdesc.Output(framework::GradVarName("Y")).size() > 0) {
    auto Y_grad = opdesc.Output(framework::GradVarName("Y")).front();
    param_.Y_grad = GetMutableVar<Tensor>(scope, Y_grad);
  }
  param_.Y = GetVar<lite::Tensor>(scope, Y_name);
  param_.Out_grad = GetVar<lite::Tensor>(scope, Out_name);
  param_.X_grad = GetMutableVar<lite::Tensor>(scope, X_grad);
  param_.axis = opdesc.GetAttr<int>("axis");

  return true;
}

#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(elementwise_sub, paddle::lite::operators::ElementwiseOp);
#ifdef LITE_WITH_X86
REGISTER_LITE_OP(elementwise_sub_grad,
                 paddle::lite::operators::ElementwiseGradExplicitOp);
#endif
REGISTER_LITE_OP(elementwise_add, paddle::lite::operators::ElementwiseOp);
