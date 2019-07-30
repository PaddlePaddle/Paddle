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

#include "paddle/fluid/lite/operators/relu_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ReluOp::CheckShape() const { return true; }
bool ReluOp::InferShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  // TODO(Superjomn) Enable data sharing.
  param_.Out->Resize(param_.X->dims());
  // share lod
  param_.Out->raw_tensor().set_lod(param_.X->lod());
  return true;
}

bool ReluOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  CHECK(param_.X);
  CHECK(param_.Out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(relu, paddle::lite::operators::ReluOp);
