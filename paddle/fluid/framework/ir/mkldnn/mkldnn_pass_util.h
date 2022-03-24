// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

void SaveQuantInfo(
    ir::Graph* graph, const std::string& flag, const std::string key_suffix,
    std::unordered_map<std::string, std::vector<float>>* var_quant_scales) {
  VLOG(3) << "save variables in the first op's attr";

  for (auto* op_node :
       ir::TopologyVarientSort(*graph, static_cast<ir::SortKind>(0))) {
    if (!op_node->IsOp() || op_node->Op()->Type() == "feed" ||
        op_node->Op()->Type() == "fetch")
      continue;
    op_node->Op()->SetAttr(flag, true);

    for (auto iter = var_quant_scales->begin(); iter != var_quant_scales->end();
         ++iter) {
      op_node->Op()->SetAttr(iter->first + key_suffix, iter->second);
    }
    break;
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
