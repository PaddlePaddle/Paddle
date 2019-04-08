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

#include "paddle/fluid/inference/anakin/convert/detection_out.h"
#include <algorithm>
#include <map>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;

namespace paddle {
namespace inference {
namespace anakin {

void DetectionOutOpConverter::operator()(const framework::proto::OpDesc &op,
                                         const framework::BlockDesc &block_desc,
                                         const framework::Scope &scope,
                                         bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  auto target_name = op_desc.Input("TargetBox").front();
  auto prior_box_name = op_desc.Input("PriorBox").front();
  auto scores_name = op_desc.Input("Scores").front();
  auto output_name = op_desc.Output("Out").front();

  auto op_name = op_desc.Type() + ":" + op_desc.Output("Out").front();

  auto code_type = boost::get<std::string>(op_desc.GetAttr("code_type"));
  auto background_label = boost::get<int>(op_desc.GetAttr("background_label"));
  auto score_threshold = boost::get<float>(op_desc.GetAttr("score_threshold"));
  auto nms_top_k = boost::get<int>(op_desc.GetAttr("nms_top_k"));
  auto nms_threshold = boost::get<float>(op_desc.GetAttr("nms_threshold"));
  auto nms_eta = boost::get<float>(op_desc.GetAttr("nms_eta"));
  auto keep_top_k = boost::get<int>(op_desc.GetAttr("keep_top_k"));
  std::string anakin_code_type;
  if (code_type == "decode_center_size") {
    anakin_code_type = "CENTER_SIZE";
  } else if (code_type == "encode_center_size") {
    PADDLE_THROW(
        "Not support encode_center_size code_type in DetectionOut of anakin");
  }

  engine_->AddOp(op_name, "DetectionOutput",
                 {target_name, scores_name, prior_box_name}, {output_name});
  engine_->AddOpAttr(op_name, "share_location", true);
  engine_->AddOpAttr(op_name, "variance_encode_in_target", false);
  engine_->AddOpAttr(op_name, "class_num", static_cast<int>(0));
  engine_->AddOpAttr(op_name, "background_id", background_label);
  engine_->AddOpAttr(op_name, "keep_top_k", keep_top_k);
  engine_->AddOpAttr(op_name, "code_type", anakin_code_type);
  engine_->AddOpAttr(op_name, "conf_thresh", score_threshold);
  engine_->AddOpAttr(op_name, "nms_top_k", nms_top_k);
  engine_->AddOpAttr(op_name, "nms_thresh", nms_threshold);
  engine_->AddOpAttr(op_name, "nms_eta", nms_eta);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(detection_out, DetectionOutOpConverter);
