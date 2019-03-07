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

#include "paddle/fluid/inference/anakin/convert/flatten.h"
#include <vector>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;
using anakin::PTuple;

namespace paddle {
namespace inference {
namespace anakin {

void FlattenOpConverter::operator()(const framework::proto::OpDesc &op,
                                    const framework::Scope &scope,
                                    bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1UL);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1UL);

  auto input = op_desc.Input("X").front();
  auto output = op_desc.Output("Out").front();
  int axis = boost::get<int>(op_desc.GetAttr("axis"));
  PADDLE_ENFORCE(axis == 1,
                 "the anakin flatten op converter now only support aixs == 1.");

  std::vector<int> out_dims = {0, -1, 1, 1};
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();
  engine_->AddOp(op_name, "Reshape", {input}, {output});
  engine_->AddOpAttr<PTuple<int>>(op_name, "dims", out_dims);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(flatten, FlattenOpConverter);
