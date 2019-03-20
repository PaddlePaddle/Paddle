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

#include "paddle/fluid/inference/anakin/convert/pool2d.h"
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

void Pool2dOpConverter::operator()(const framework::proto::OpDesc &op,
                                   const framework::Scope &scope,
                                   bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
  PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

  auto x_name = op_desc.Input("X").front();
  auto y_name = op_desc.Output("Out").front();
  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  bool global_pooling = boost::get<bool>(op_desc.GetAttr("global_pooling"));
  std::string pool_type =
      boost::get<std::string>(op_desc.GetAttr("pooling_type"));
  std::vector<int> ksize =
      boost::get<std::vector<int>>(op_desc.GetAttr("ksize"));
  std::vector<int> strides =
      boost::get<std::vector<int>>(op_desc.GetAttr("strides"));
  std::vector<int> paddings =
      boost::get<std::vector<int>>(op_desc.GetAttr("paddings"));
  bool ceil_mode = boost::get<bool>(op_desc.GetAttr("ceil_mode"));
  std::string anakin_pool_type;
  if (pool_type == "max") {
    anakin_pool_type = "MAX";
  } else if (pool_type == "avg") {
    if (paddings[0] || paddings[1]) {
      anakin_pool_type = "AVGEXC";
    } else {
      anakin_pool_type = "AVG";
    }
  } else {
    PADDLE_THROW("TensorRT unsupported pooling type!");
  }

  engine_->AddOp(op_name, "Pooling", {x_name}, {y_name});
  engine_->AddOpAttr<PTuple<int>>(op_name, "pool_size", ksize);
  engine_->AddOpAttr<PTuple<int>>(op_name, "strides", strides);
  engine_->AddOpAttr<PTuple<int>>(op_name, "padding", paddings);
  engine_->AddOpAttr(op_name, "method", anakin_pool_type);
  engine_->AddOpAttr(op_name, "global_pooling", global_pooling);
  engine_->AddOpAttr(op_name, "cmp_out_shape_floor_as_conv", !ceil_mode);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(pool2d, Pool2dOpConverter);
