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

#include "paddle/fluid/lite/operators/uniform_random_op.h"
#include "paddle/fluid/lite/core/op_lite.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool UniformRandomOpLite::CheckShape() const { return true; }

bool UniformRandomOpLite::InferShape() const {
  param_.Out->Resize(param_.shape);
  return true;
}

bool UniformRandomOpLite::AttachImpl(const cpp::OpDesc& opdesc,
                                     lite::Scope* scope) {
  param_.shape = opdesc.GetAttr<std::vector<int64_t>>("shape");
  param_.min = opdesc.GetAttr<float>("min");
  param_.max = opdesc.GetAttr<float>("max");
  param_.seed = opdesc.GetAttr<int>("seed");
  param_.dtype = opdesc.GetAttr<int>("dtype");
  param_.Out = GetMutableVar<Tensor>(scope, opdesc.Output("Out").front());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(uniform_random, paddle::lite::operators::UniformRandomOpLite);
