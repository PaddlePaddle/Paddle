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

#include "paddle/fluid/inference/anakin/convert/split.h"
#include <algorithm>
#include <vector>

using anakin::PTuple;

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void SplitOpConverter<TargetT>::operator()(
    const framework::proto::OpDesc &op, const framework::BlockDesc &block_desc,
    const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  auto input_name = op_desc.Input("X").front();
  auto y_names = op_desc.Output("Out");
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();
  int axis = boost::get<int>(op_desc.GetAttr("axis"));

  std::vector<int> output_lengths =
      boost::get<std::vector<int>>(op_desc.GetAttr("sections"));

  int split_num = output_lengths.size();
  PADDLE_ENFORCE(split_num > 1,
                 "anakin split op converter: the split num should > 1");
  int num_sum = 0;
  std::vector<int> slice_point;
  for (int i = 0; i < split_num - 1; i++) {
    num_sum += output_lengths[i];
    slice_point.push_back(num_sum);
  }
  this->engine_->AddOp(op_name, "Slice", {input_name}, y_names);
  this->engine_->AddOpAttr(op_name, "axis", axis);
  this->engine_->template AddOpAttr<PTuple<int>>(op_name, "slice_point",
                                                 slice_point);
  // slice_dim is useless in anakin
  this->engine_->AddOpAttr(op_name, "slice_dim", 4);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
#ifdef PADDLE_WITH_CUDA
REGISTER_CUDA_ANAKIN_OP_CONVERTER(split, SplitOpConverter<::anakin::saber::NV>);
#endif

REGISTER_CPU_ANAKIN_OP_CONVERTER(split, SplitOpConverter<::anakin::saber::X86>);
