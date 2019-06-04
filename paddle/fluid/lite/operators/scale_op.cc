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

#include "paddle/fluid/lite/operators/scale_op.h"
#include "paddle/fluid/lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool ScaleOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  return true;
}

bool ScaleOp::InferShape() const {
  param_.output->Resize(param_.x->dims());
  return true;
}

bool ScaleOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto output = op_desc.Output("Out").front();
  param_.x = scope->FindVar(x)->GetMutable<Tensor>();
  param_.output = scope->FindVar(output)->GetMutable<Tensor>();
  param_.scale = op_desc.GetAttr<float>("scale");
  param_.bias = op_desc.GetAttr<float>("bias");
  param_.bias_after_scale = op_desc.GetAttr<bool>("bias_after_scale");
  CHECK(param_.x);
  CHECK(param_.output);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(scale, paddle::lite::operators::ScaleOp);
