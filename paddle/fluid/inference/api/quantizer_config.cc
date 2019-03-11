// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/paddle_quantizer_config.h"

namespace paddle {

QuantizerConfig::QuantizerConfig() {
  // The default configuration of scale computing algorightms
  rules_["conv2d"]["Input"] = ScaleAlgo::KL;
  rules_["conv2d"]["Filter"] = ScaleAlgo::MAX_CH;
  rules_["conv2d"]["Bias"] = ScaleAlgo::NONE;  // do not compute scale
  rules_["conv2d"]["ResidualData"] = ScaleAlgo::KL;
  rules_["conv2d"]["Output"] = ScaleAlgo::KL;  // do not compute scale

  rules_["pool2d"]["X"] = ScaleAlgo::KL;
  rules_["pool2d"]["Out"] = ScaleAlgo::KL;  // do not compute scale
}

ScaleAlgo QuantizerConfig::scale_algo(const std::string& op_type_name,
                                      const std::string& conn_name) const {
  if (rules_.find(op_type_name) != rules_.end()) {
    auto op_rule = rules_.at(op_type_name);
    if (op_rule.find(conn_name) != op_rule.end()) return op_rule.at(conn_name);
  }
  return default_scale_algo_;
}

}  // namespace paddle
