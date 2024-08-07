// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/quantize_helper.h"

namespace paddle::framework::ir {

void SaveQuantInfoInTheGraph(
    ir::Graph* graph,
    const std::string& flag,
    const std::string& key_suffix,
    const std::unordered_map<std::string, std::vector<float>>& info_map) {
  const std::string suffix = "_" + key_suffix + "_" + flag;
  if (!graph->Has(flag)) {
    graph->Set(flag, new bool(true));
  }
  for (const auto& iter : info_map) {
    graph->Set(iter.first + suffix, new std::vector<float>(iter.second));
  }
}

std::unordered_map<std::string, std::vector<float>> GetQuantInfoFromTheGraph(
    ir::Graph* graph, const std::string& flag, const std::string& key_suffix) {
  std::unordered_map<std::string, std::vector<float>> info_map;
  const std::string suffix = "_" + key_suffix + "_" + flag;
  if (graph->Has(flag)) {
    std::vector<std::string> attr_names = graph->AttrNames();
    for (auto const& fake_name : attr_names) {
      size_t pos = fake_name.find(suffix);
      if (pos != std::string::npos) {
        std::string name = fake_name.substr(0, pos);
        auto scales_vector = graph->Get<std::vector<float>>(fake_name);
        info_map.insert(std::make_pair(name, scales_vector));
      }
    }
  }
  return info_map;
}

bool AreScalesPresentForNodes(
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
    std::initializer_list<Node*> nodes) {
  bool present = true;
  for (auto node : nodes) {
    if (var_quant_scales->count(node->Name()) == 0) {
      present = false;
    }
  }
  return present;
}

float GetScaleValueForNode(
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
    Node* node) {
  return var_quant_scales->at(node->Name())[0];
}

std::vector<float> GetScaleVecValueForNode(
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
    Node* node) {
  return var_quant_scales->at(node->Name());
}

}  // namespace paddle::framework::ir
