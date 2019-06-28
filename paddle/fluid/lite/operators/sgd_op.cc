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

#include "paddle/fluid/lite/operators/sgd_op.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SGDOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.Param);
  CHECK_OR_FALSE(param_.LearningRate);
  CHECK_OR_FALSE(param_.Grad);
  CHECK_OR_FALSE(param_.ParamOut);
  return true;
}

bool SGDOpLite::InferShape() const {
  auto lr_dims = param_.LearningRate->dims().data();
#ifndef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
  CHECK_EQ_OR_FALSE(framework::product(lr_dims), 1);
#endif
  param_.ParamOut->Resize(param_.Param->dims());
  return true;
}

bool SGDOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto Param_name = opdesc.Input("Param").front();
  auto LearningRate_name = opdesc.Input("LearningRate").front();
  auto Grad_name = opdesc.Input("Grad").front();
  auto ParamOut_name = opdesc.Output("ParamOut").front();

  param_.Param = GetVar<lite::Tensor>(scope, Param_name);
  param_.LearningRate = GetVar<lite::Tensor>(scope, LearningRate_name);
  param_.Grad = GetVar<Tensor>(scope, Grad_name);
  param_.ParamOut = GetMutableVar<Tensor>(scope, ParamOut_name);

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sgd, paddle::lite::operators::SGDOpLite);
