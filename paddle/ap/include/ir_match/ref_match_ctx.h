// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <unordered_map>
#include <vector>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/ir_match/ref_node_info.h"

namespace ap::ir_match {

template <typename IrValueT, typename IrOperandT>
struct RefMatchCtxImpl {
  std::unordered_map<IrValueT, std::vector<RefNodeInfo<IrValueT, IrOperandT>>>
      value2ref_node_info;
  std::unordered_map<IrOperandT, RefNodeInfo<IrValueT, IrOperandT>>
      operand2node_info;

  adt::Result<adt::Ok> AddRefNodeInfo(
      const RefNodeInfo<IrValueT, IrOperandT>& node_info) {
    auto* vec = &value2ref_node_info[node_info->ir_value];
    vec->emplace_back(node_info);
    for (const auto& op_operand : *node_info->op_operands_subset) {
      ADT_CHECK(operand2node_info.emplace(op_operand, node_info).second);
    }
    return adt::Ok{};
  }
};

template <typename IrValueT, typename IrOperandT>
ADT_DEFINE_RC(RefMatchCtx, RefMatchCtxImpl<IrValueT, IrOperandT>);

}  // namespace ap::ir_match
