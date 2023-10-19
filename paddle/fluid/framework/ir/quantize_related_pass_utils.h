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

static inline void SaveInfoInTheTmpOp(
    ir::Graph* graph,
    const std::string& flag,
    const std::string& key_suffix,
    const std::unordered_map<std::string, std::vector<float>>& info_map) {
  VLOG(3) << "save variables in the first op's attr";

  const std::string suffix = "_" + key_suffix + "_" + flag;
  OpDesc op_desc;
  op_desc.SetType("save");
  auto* op_node = graph->CreateOpNode(&op_desc);

  op_node->Op()->SetAttr(flag, true);
  for (auto iter = info_map.begin(); iter != info_map.end(); ++iter) {
    op_node->Op()->SetAttr(iter->first + suffix, iter->second);
  }
}

static inline void SaveQuantInfoInTheGraph(
    ir::Graph* graph,
    const std::string& flag,
    const std::string& key_suffix,
    const std::unordered_map<std::string, std::vector<float>>& info_map) {
  VLOG(1) << "Save quant info in the graph!";
  const std::string suffix = "_" + key_suffix + "_" + flag;
  graph->Set(flag, new bool(true));
  for (auto iter = info_map.begin(); iter != info_map.end(); ++iter) {
    VLOG(1) << "SaveQuantInfoInTheGraph set attr: " << iter->first + suffix;
    graph->Set(iter->first + suffix, new std::vector<float>(iter->second));
  }
}

static void GetInfoFromTheTmpOp(
    ir::Graph* graph,
    const std::string& flag,
    const std::string& key_suffix,
    std::unordered_map<std::string, std::vector<float>>* info_map) {
  VLOG(3) << "get variables from the first op's attr";

  const std::string suffix = "_" + key_suffix + "_" + flag;
  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp() || op_node->Op()->Type() != "save") continue;
    VLOG(1) << "Come in save op";
    auto* op_desc = op_node->Op();
    if (op_desc->GetAttrIfExists<bool>(flag)) {
      VLOG(1) << "flag is true";
      op_desc->RemoveAttr(flag);
      std::vector<std::string> attr_names = op_desc->AttrNames();
      VLOG(1) << "attr_names size:" << attr_names.size();
      for (auto fake_name : attr_names) {
        VLOG(1) << "fake_name:" << fake_name;
        size_t pos = fake_name.find(suffix);
        if (pos != std::string::npos) {
          std::string name = fake_name.substr(0, pos);
          VLOG(1) << "name:" << name;
          auto scales_vector =
              PADDLE_GET_CONST(std::vector<float>, op_desc->GetAttr(fake_name));
          VLOG(1) << "scales_vector:" << scales_vector[0];
          info_map->insert(std::make_pair(name, scales_vector));
          VLOG(1) << "insert success:";
          op_desc->RemoveAttr(fake_name);
          VLOG(1) << "remove success:";
        }
      }
      graph->RemoveNode(op_node);
      VLOG(1) << "remove op node success:";
      break;
    }
  }
}

static inline void GetQuantInfoFromTheGraph(
    ir::Graph* graph,
    const std::string& flag,
    const std::string& key_suffix,
    std::unordered_map<std::string, std::vector<float>>* info_map) {
  VLOG(1) << "Get quant info from the graph attrs!";
  const std::string suffix = "_" + key_suffix + "_" + flag;
  VLOG(1) << "flag:" << (graph->Has(flag) ? 1 : 0);
  if (graph->Has(flag)) {
    std::vector<std::string> attr_names = graph->AttrNames();
    VLOG(1) << "attr_names size:" << attr_names.size();
    for (auto fake_name : attr_names) {
      VLOG(1) << "fake_name:" << fake_name;
      size_t pos = fake_name.find(suffix);
      if (pos != std::string::npos) {
        std::string name = fake_name.substr(0, pos);
        VLOG(1) << "name:" << name;
        auto scales_vector = graph->Get<std::vector<float>>(fake_name);
        VLOG(1) << "scales_vector:" << scales_vector[0];
        info_map->insert(std::make_pair(name, scales_vector));
      }
    }
  }
}

static inline std::unordered_map<std::string, std::vector<float>>
GetQuantInfoFromTheGraph(ir::Graph* graph,
                         const std::string& flag,
                         const std::string& key_suffix) {
  std::unordered_map<std::string, std::vector<float>> info_map;
  VLOG(1) << "Get quant info from the graph attrs!";
  const std::string suffix = "_" + key_suffix + "_" + flag;
  VLOG(1) << "flag:" << (graph->Has(flag) ? 1 : 0);
  if (graph->Has(flag)) {
    std::vector<std::string> attr_names = graph->AttrNames();
    VLOG(1) << "attr_names size:" << attr_names.size();
    for (auto fake_name : attr_names) {
      VLOG(1) << "fake_name:" << fake_name;
      size_t pos = fake_name.find(suffix);
      if (pos != std::string::npos) {
        std::string name = fake_name.substr(0, pos);
        VLOG(1) << "name:" << name;
        auto scales_vector = graph->Get<std::vector<float>>(fake_name);
        VLOG(1) << "scales_vector:" << scales_vector[0];
        info_map.insert(std::make_pair(name, scales_vector));
      }
    }
  }
  return info_map;
}

static inline bool AreScalesPresentForNodes(
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

static inline float GetScaleValueForNode(
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales,
    Node* node) {
  return var_quant_scales->at(node->Name())[0];
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
