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

#include "paddle/fluid/lite/operators/sequence_reshape_op.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequenceReshapeOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  auto x_dims = param_.x->dims();
  CHECK_EQ_OR_FALSE(x_dims.size(), 2U);
  return true;
}

bool SequenceReshapeOp::InferShape() const {
  int new_dim = param_.new_dim;
  auto x_numel = param_.x->dims().production();
  std::vector<int64_t> out_shape{x_numel / new_dim,
                                 static_cast<int64_t>(new_dim)};
  param_.output->Resize(lite::DDim(out_shape));
  return true;
}

bool SequenceReshapeOp::AttachImpl(const cpp::OpDesc &opdesc,
                                   lite::Scope *scope) {
  param_.x =
      scope->FindVar(opdesc.Input("X").front())->GetMutable<lite::Tensor>();
  param_.output =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();

  param_.new_dim = opdesc.GetAttr<int>("new_dim");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_reshape, paddle::lite::operators::SequenceReshapeOp);
