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

#pragma once

#include <string>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

void SaveQuantInfoInTheGraph(
    ir::Graph* graph,
    const std::string& flag,
    const std::string& key_suffix,
    const std::unordered_map<std::string, std::vector<float>>& info_map);

std::unordered_map<std::string, std::vector<float>> GetQuantInfoFromTheGraph(
    ir::Graph* graph, const std::string& flag, const std::string& key_suffix);

bool AreScalesPresentForNodes(
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
    std::initializer_list<Node*> nodes);

float GetScaleValueForNode(
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
    Node* node);

std::vector<float> GetScaleVecValueForNode(
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
    Node* node);

template <typename T>
inline std::string Vec2Str(const std::vector<T>& vec) {
  std::ostringstream os;
  if (vec.empty()) {
    os << "()";
    return os.str();
  }
  os << "(";
  for (size_t i = 0; i < vec.size() - 1; ++i) {
    os << vec[i] << ",";
  }
  os << vec[vec.size() - 1] << ")";
  return os.str();
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
