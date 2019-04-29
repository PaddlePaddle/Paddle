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
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  // TODO(Superjomn) Enable data sharing.
  param_.output->Resize(param_.input->dims());
  // param_.output->ShareDataWith(*param_.input);
  // share lod
  // param_.output->set_lod(param_.input->lod());
  return true;
}

bool ReluOp::AttachImpl(const OpDesc &opdesc, lite::Scope *scope) {
  param_.input = const_cast<Tensor *>(
      &scope->FindVar(opdesc.Input("Input").front())->Get<Tensor>());
  param_.output =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<Tensor>();
  CHECK(param_.input);
  CHECK(param_.output);
  kernel_->SetParam(param_);
  return true;
}

REGISTER_LITE_OP(relu, ReluOp);

}  // namespace operators
}  // namespace lite
}  // namespace paddle
