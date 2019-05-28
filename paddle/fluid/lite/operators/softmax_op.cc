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

#include "paddle/fluid/lite/operators/softmax_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SoftmaxOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  auto dim_x = param_.x->dims();
  auto rank_x = dim_x.size();
  CHECK_OR_FALSE(param_.axis >= -rank_x && param_.axis < rank_x);
  return true;
}

bool SoftmaxOp::InferShape() const {
  param_.output->Resize(param_.x->dims());
  return true;
}

bool SoftmaxOp::AttachImpl(const OpDesc &opdesc, lite::Scope *scope) {
  param_.x = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.output =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  param_.axis = GetAttr<int>(opdesc.GetAttr("axis"));
  CHECK(param_.x);
  CHECK(param_.output);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(softmax, paddle::lite::operators::SoftmaxOp);
