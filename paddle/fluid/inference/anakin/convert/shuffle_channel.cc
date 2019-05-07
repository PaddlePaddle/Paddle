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

#include "paddle/fluid/inference/anakin/convert/shuffle_channel.h"
#include <algorithm>
#include <string>
#include <vector>

using anakin::PTuple;

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, ::anakin::Precision PrecisionT>
void ShuffleChannelOpConverter<TargetT, PrecisionT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto input = op_desc.Input("X").front();
  auto output = op_desc.Output("Out").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();
  this->engine_->AddOp(op_name, "ShuffleChannel", {input}, {output});

  auto group = boost::get<int>(op_desc.GetAttr("group"));
  this->engine_->AddOpAttr(op_name, "group", group);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(shuffle_channel, ShuffleChannelOpConverter);
