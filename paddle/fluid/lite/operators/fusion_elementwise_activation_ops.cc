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

#include "paddle/fluid/lite/operators/fusion_elementwise_activation_ops.h"
#include <string>
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FusionElementwiseActivationOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool FusionElementwiseActivationOp::InferShape() const {
  CHECK_OR_FALSE(param_.X->dims().size() >= param_.Y->dims().size());
  param_.Out->Resize(param_.X->dims());
  return true;
}

bool FusionElementwiseActivationOp::AttachImpl(const cpp::OpDesc& opdesc,
                                               lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Y_name = opdesc.Input("Y").front();
  auto Out_name = opdesc.Output("Out").front();

  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Y = GetVar<lite::Tensor>(scope, Y_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);
  param_.axis = opdesc.GetAttr<int>("axis");
  param_.act_type = opdesc.GetAttr<std::string>("act_type");
  // TODO(sangoly): support more activation types.
  CHECK(param_.act_type == "relu") << "Only relu activation be supported now";

  return true;
}

#ifdef LITE_WITH_X86
bool FusionElementwiseActivationGradExplicitOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Y);
  CHECK_OR_FALSE(param_.X_grad);
  CHECK_OR_FALSE(param_.Y_grad);
  CHECK_OR_FALSE(param_.Out_grad);
  return true;
}

bool FusionElementwiseActivationGradExplicitOp::InferShape() const {
  param_.X_grad->Resize(param_.Out_grad->dims());
  param_.Y_grad->Resize(param_.Y->dims());
  return true;
}

bool FusionElementwiseActivationGradExplicitOp::AttachImpl(
    const cpp::OpDesc& opdesc, lite::Scope* scope) {
  CHECK_EQ(opdesc.InputArgumentNames().size(), 1UL);
  auto Out_name = opdesc.Input(framework::GradVarName("Out")).front();
  auto X_name = opdesc.Output(framework::GradVarName("X")).front();
  auto Y_name = opdesc.Output(framework::GradVarName("Y")).front();

  param_.Out_grad = GetVar<lite::Tensor>(scope, Out_name);
  param_.X_grad = GetMutableVar<lite::Tensor>(scope, X_name);
  param_.Y_grad = GetMutableVar<Tensor>(scope, Y_name);
  param_.axis = opdesc.GetAttr<int>("axis");
  param_.act_type = opdesc.GetAttr<std::string>("act_type");
  // TODO(sangoly): support more activation types.
  CHECK(param_.act_type == "relu") << "Only relu activation be supported now";

  return true;
}
#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fusion_elementwise_sub_activation,
                 paddle::lite::operators::FusionElementwiseActivationOp);
#ifdef LITE_WITH_X86
REGISTER_LITE_OP(
    fusion_elementwise_sub_activation_grad,
    paddle::lite::operators::FusionElementwiseActivationGradExplicitOp);
#endif
REGISTER_LITE_OP(fusion_elementwise_add_activation,
                 paddle::lite::operators::FusionElementwiseActivationOp);
