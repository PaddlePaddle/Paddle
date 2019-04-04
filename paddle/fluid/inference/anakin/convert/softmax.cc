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

#include "paddle/fluid/inference/anakin/convert/softmax.h"

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

void SoftMaxOpConverter::operator()(const framework::proto::OpDesc &op,
                                    const framework::BlockDesc &block_desc,
                                    const framework::Scope &scope,
                                    bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1UL);

  auto input = op_desc.Input("X").front();
  auto output = op_desc.Output("Out").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  auto input_var_desc = block_desc.FindVar(input);
  PADDLE_ENFORCE(input_var_desc,
                 "Cant find %s variable When runing Anakin Softmax converter.",
                 input);
  auto input_shape_in_fluid = input_var_desc->GetShape();
  size_t input_dims = input_shape_in_fluid.size();

  engine_->AddOp(op_name, "Softmax", {input}, {output});
  engine_->AddOpAttr(op_name, "axis", static_cast<int>(input_dims - 1));
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(softmax, SoftMaxOpConverter);
