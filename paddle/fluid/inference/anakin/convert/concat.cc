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

#include "paddle/fluid/inference/anakin/convert/concat.h"
#include <algorithm>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::Precision;
using anakin::saber::NV;
using anakin::saber::X86;
using anakin::saber::Shape;
using anakin::PBlock;
using anakin::PTuple;

namespace paddle {
namespace inference {
namespace anakin {

void ConcatOpConverter::operator()(const framework::proto::OpDesc &op,
                                   const framework::BlockDesc &block_desc,
                                   const framework::Scope &scope,
                                   bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  int axis = boost::get<int>(op_desc.GetAttr("axis"));
  auto input_names = op_desc.Input("X");
  // PADDLE_ENFORCE(axis > 0,
  //               "The axis attr of Concat op should be large than 0 for trt");

  auto y_name = op_desc.Output("Out").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  engine_->AddOp(op_name, "Concat", input_names, {y_name});
  engine_->AddOpAttr(op_name, "axis", axis);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(concat, ConcatOpConverter);
