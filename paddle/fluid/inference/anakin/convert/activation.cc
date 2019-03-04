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

#include "paddle/fluid/inference/anakin/convert/activation.h"
#include <algorithm>
#include <map>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

ActivationOpConverter::ActivationOpConverter(const std::string &op_type)
    : op_type_(op_type) {
  auto it = anakin_op_types_.find(op_type_);
  PADDLE_ENFORCE(it != anakin_op_types_.end(),
                 "activation op type is not support");
  anakin_op_type_ = it->second;
}

void ActivationOpConverter::operator()(const framework::proto::OpDesc &op,
                                       const framework::Scope &scope,
                                       bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();
  auto input_name = op_desc.Input("X").front();
  auto output_name = op_desc.Output("Out").front();
  engine_->AddOp(op_name, "Activation", {input_name}, {output_name});
  engine_->AddOpAttr(op_name, "type", anakin_op_type_);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(sigmoid, SigmoidOpConverter);
REGISTER_ANAKIN_OP_CONVERTER(tanh, TanhOpConverter);
