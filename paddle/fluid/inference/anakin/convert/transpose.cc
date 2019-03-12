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

#include "paddle/fluid/inference/anakin/convert/transpose.h"
#include <algorithm>
#include <string>
#include <vector>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;
using anakin::PTuple;

namespace paddle {
namespace inference {
namespace anakin {

void TransposeOpConverter::operator()(const framework::proto::OpDesc &op,
                                      const framework::Scope &scope,
                                      bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto input = op_desc.Input("X").front();
  auto output = op_desc.Output("Out").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();
  engine_->AddOp(op_name, "Permute", {input}, {output});

  auto axis = boost::get<std::vector<int>>(op_desc.GetAttr("axis"));
  size_t axis_size = axis.size();
  while (axis.size() < 4) {
    axis.push_back(axis_size);
    axis_size += 1;
  }
  engine_->AddOpAttr<PTuple<int>>(op_name, "dims", axis);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(transpose, TransposeOpConverter);
