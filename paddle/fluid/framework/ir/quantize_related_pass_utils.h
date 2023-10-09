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

namespace paddle {
namespace framework {
namespace ir {

static void SaveInfoInTheTmpOp(
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
    VLOG(5) << "Come in save op";
    auto* op_desc = op_node->Op();
    if (op_desc->GetAttrIfExists<bool>(flag)) {
      VLOG(5) << "flag is true";
      op_desc->RemoveAttr(flag);
      std::vector<std::string> attr_names = op_desc->AttrNames();
      for (auto fake_name : attr_names) {
        VLOG(5) << "fake_name:" << fake_name;
        size_t pos = fake_name.find(suffix);
        if (pos != std::string::npos) {
          std::string name = fake_name.substr(0, pos);
          VLOG(5) << "name:" << name;
          auto scales_vector =
              PADDLE_GET_CONST(std::vector<float>, op_desc->GetAttr(fake_name));
          VLOG(5) << "scales_vector:" << scales_vector[0];
          info_map->insert(std::make_pair(name, scales_vector));
          VLOG(5) << "insert success:";
          op_desc->RemoveAttr(fake_name);
          VLOG(5) << "remove success:";
        }
      }
      graph->RemoveNode(op_node);
      VLOG(5) << "remove op node success:";
      break;
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
