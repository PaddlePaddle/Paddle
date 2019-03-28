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

#include "paddle/fluid/inference/anakin/convert/density_prior_box.h"
#include <algorithm>
#include <map>
#include <vector>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;
using anakin::PTuple;

namespace paddle {
namespace inference {
namespace anakin {

void DensityPriorBoxOpConverter::operator()(
    const framework::proto::OpDesc& op, const framework::BlockDesc& block_desc,
    const framework::Scope& scope, bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  auto input_name = op_desc.Input("Input").front();
  auto image_name = op_desc.Input("Image").front();
  auto output_name = op_desc.Output("Boxes").front();

  auto op_name = op_desc.Type() + ":" + op_desc.Output("Boxes").front();

  auto fixed_sizes =
      boost::get<std::vector<float>>(op_desc.GetAttr("fixed_sizes"));
  auto fixed_ratios =
      boost::get<std::vector<float>>(op_desc.GetAttr("fixed_ratios"));
  auto densities = boost::get<std::vector<int>>(op_desc.GetAttr("densities"));
  std::vector<float> dens;
  for (auto& ele : densities) {
    dens.push_back(static_cast<float>(ele));
  }

  // lack flip
  // auto clip = boost::get<bool>(op_desc.GetAttr("clip"));
  auto variances = boost::get<std::vector<float>>(op_desc.GetAttr("variances"));
  for (auto& ele : variances) {
    LOG(INFO) << ele;
  }

  // lack img_h, img_w
  auto step_h = boost::get<float>(op_desc.GetAttr("step_h"));
  auto step_w = boost::get<float>(op_desc.GetAttr("step_w"));
  auto offset = boost::get<float>(op_desc.GetAttr("offset"));
  PTuple<std::string> t_order;
  t_order.push_back("MIN");
  t_order.push_back("COM");
  t_order.push_back("MAX");

  std::vector<float> temp_v = {};

  engine_->AddOp(op_name, "PriorBox", {input_name, image_name}, {output_name});
  engine_->AddOpAttr<PTuple<float>>(op_name, "min_size", temp_v);
  engine_->AddOpAttr<PTuple<float>>(op_name, "max_size", temp_v);
  engine_->AddOpAttr<PTuple<float>>(op_name, "aspect_ratio", temp_v);
  engine_->AddOpAttr<PTuple<float>>(op_name, "fixed_size", fixed_sizes);
  engine_->AddOpAttr<PTuple<float>>(op_name, "fixed_ratio", fixed_ratios);
  engine_->AddOpAttr<PTuple<float>>(op_name, "density", dens);
  engine_->AddOpAttr(op_name, "is_flip", static_cast<bool>(false));
  engine_->AddOpAttr(op_name, "is_clip", static_cast<bool>(false));
  engine_->AddOpAttr<PTuple<float>>(op_name, "variance", variances);
  engine_->AddOpAttr(op_name, "img_h", static_cast<int>(0));
  engine_->AddOpAttr(op_name, "img_w", static_cast<int>(0));
  engine_->AddOpAttr(op_name, "step_h", step_h);
  engine_->AddOpAttr(op_name, "step_w", step_w);
  engine_->AddOpAttr(op_name, "offset", offset);
  engine_->AddOpAttr<PTuple<std::string>>(op_name, "order", t_order);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(density_prior_box, DensityPriorBoxOpConverter);
