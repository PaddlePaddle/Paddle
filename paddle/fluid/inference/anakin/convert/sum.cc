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

#include "paddle/fluid/inference/anakin/convert/sum.h"
#include <algorithm>
#include <string>
#include <vector>

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

void SumOpConverter::operator()(const framework::proto::OpDesc &op,
                                const framework::Scope &scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 2);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto input_names = op_desc.Input("X");
  auto out_name = op_desc.Output("Out").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  std::vector<float> coeff = {1, 1};
  std::string elementwise_type = "Add";
  engine_->AddOp(op_name, "Eltwise", input_names, {out_name});
  engine_->AddOpAttr<PTuple<float>>(op_name, "coeff", coeff);
  engine_->AddOpAttr<std::string>(op_name, "type", elementwise_type);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(sum, SumOpConverter);
