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
#include <algorithm>

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
  auto in_dims = scope.FindVar(input)->Get<framework::LoDTensor>().dims();
  //LOG(INFO) << "========= in_dmis: size[" << framework::vectorize2int(in_dims).size() << "]";
  int axis = boost::get<int>(op_desc.GetAttr("axis"));

  int inner = 1;
  int outer = 1;
  for (int i = 0; i < in_dims.size(); i++) {
    if (i < axis) {
      outer *= in_dims[i];
    } else {
      inner *= in_dims[i];
    }
  }

  std::vector<int> out_dims = {1, outer, inner, 1};
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();
  engine_->AddOp(op_name, "Reshape", {input}, {output});

  //engine_->AddOpAttr(op_name, "axis", 1);
  //engine_->AddOpAttr(op_name, "num_axes", 1);
  engine_->AddOpAttr<PTuple<int>>(op_name, "dims", out_dims);
  //engine_->AddOpAttr(op_name, "layout", "HW");
  //engine_->AddOpAttr(op_name, "layout", "NCHW");

  //int axis = boost::get<int>(op_desc.GetAttr("axis"));
  //auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();
  //auto op1 = op_name + "_1";
  //auto op1_output = op1 + ":out";
  //engine_->AddOp(op1, "Flatten", {input}, {op1_output});
  //engine_->AddOpAttr(op1, "start_axis", axis);
  //engine_->AddOpAttr(op1, "end_axis", -1);

  //auto op2 = op_name + "_2";
  //auto op2_input = op1_output;
  //engine_->AddOp(op2, "Flatten", {op2_input}, {output});
  ////engine_->AddOpAttr(op2, "input_shape", )
  //engine_->AddOpAttr(op2, "start_axis", 0);
  //engine_->AddOpAttr(op2, "end_axis", axis - 1);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(flatten, FlattenOpConverter);
