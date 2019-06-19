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

bool FusionElementwiseActivationOp::AttachImpl(const cpp::OpDesc& opdesc,
                                               lite::Scope* scope) {
  ElementwiseOp::AttachImpl(opdesc, scope);
  param_.act_type = opdesc.GetAttr<std::string>("act_type");
  // TODO(sangoly): support more activation types.
  CHECK(param_.act_type == "relu") << "Only relu activation be supported now";

  return true;
}

#ifdef LITE_WITH_X86
bool FusionElementwiseActivationGradExplicitOp::AttachImpl(
    const cpp::OpDesc& opdesc, lite::Scope* scope) {
  ElementwiseGradExplicitOp::AttachImpl(opdesc, scope);
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
