// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/anakin/convert/relu.h"
#include <algorithm>
#include <map>

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void ReluOpConverter<TargetT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();
  auto input_name = op_desc.Input("X").front();
  auto output_name = op_desc.Output("Out").front();

  this->engine_->AddOp(op_name, "ReLU", {input_name}, {output_name});
  this->engine_->AddOpAttr(op_name, "alpha", 0);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_CUDA_ANAKIN_OP_CONVERTER(relu, ReluOpConverter<::anakin::saber::NV>);
