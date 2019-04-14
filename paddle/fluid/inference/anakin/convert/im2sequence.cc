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

#include "paddle/fluid/inference/anakin/convert/im2sequence.h"
#include <algorithm>
#include <string>
#include <vector>

using anakin::PTuple;

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT, ::anakin::Precision PrecisionT>
void Im2SequenceConverter<TargetT, PrecisionT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Y").size(), 0);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto x_name = op_desc.Input("X").front();
  auto out_name = op_desc.Output("Out").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  this->engine_->AddOp(op_name, "Im2Sequence", {x_name}, {out_name});

  std::vector<int> dilations = {1, 1};
  auto paddings = boost::get<std::vector<int>>(op_desc.GetAttr("paddings"));
  auto strides = boost::get<std::vector<int>>(op_desc.GetAttr("strides"));
  auto kernels = boost::get<std::vector<int>>(op_desc.GetAttr("kernels"));

  this->engine_->template AddOpAttr<PTuple<int>>(op_name, "paddings", paddings);
  this->engine_->template AddOpAttr<PTuple<int>>(op_name, "strides", strides);
  this->engine_->template AddOpAttr<PTuple<int>>(op_name, "window_size",
                                                 kernels);
  this->engine_->template AddOpAttr<PTuple<int>>(op_name, "dilations",
                                                 dilations);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(im2sequence, Im2SequenceConverter);
