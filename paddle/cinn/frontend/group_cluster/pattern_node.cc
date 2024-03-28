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

#include "paddle/cinn/frontend/group_cluster/pattern_node.h"

namespace cinn::frontend::group_cluster {

PatternNode::PatternNode(const pir::Operation* op)
    : sink_op_(op), stmt_pattern_(ConvertToStmtPattern(op)) {}

PatternNode::PatternNode(PatternNodePtr fused_up_node,
                         PatternNodePtr fused_down_node)
    : sink_op_(fused_down_node->sink_op_),
      stmt_pattern_(MergePattern(fused_up_node->stmt_pattern_,
                                 fused_down_node->stmt_pattern_)) {
  const auto FindFromVector =
      [](std::vector<PatternNodePtr> vec,
         PatternNodePtr item) -> std::vector<PatternNodePtr>::iterator {
    return std::find(vec.begin(), vec.end(), item);
  };

  ExtendVector(&upstream_, fused_up_node->upstream_);
  ExtendVector(&upstream_, fused_down_node->upstream_);

  upstream_.erase(FindFromVector(upstream_, fused_up_node));

  ExtendVector(&downstream_, fused_up_node->downstream_);
  ExtendVector(&downstream_, fused_down_node->downstream_);
  downstream_.erase(FindFromVector(downstream_, fused_down_node));

  std::vector<PatternNodePtr>::iterator iter;
  for (const auto& upstream_node : upstream_) {
    iter = FindFromVector(upstream_node->downstream_, fused_up_node);
    if (iter != upstream_node->downstream_.end()) {
      upstream_node->downstream_.erase(iter);
    }
    iter = FindFromVector(upstream_node->downstream_, fused_down_node);
    if (iter != upstream_node->downstream_.end()) {
      upstream_node->downstream_.erase(iter);
    }
  }

  for (const auto& downstream_node : downstream_) {
    iter = FindFromVector(downstream_node->upstream_, fused_up_node);
    if (iter != downstream_node->upstream_.end()) {
      downstream_node->upstream_.erase(iter);
    }
    iter = FindFromVector(downstream_node->upstream_, fused_down_node);
    if (iter != downstream_node->upstream_.end()) {
      downstream_node->upstream_.erase(iter);
    }
  }
}

std::vector<const pir::Operation*> PatternNode::GetOps() const {
  return GetOpsInPattern(stmt_pattern_);
}

bool PatternNode::IsTrivial() const { return IsTrivialPattern(stmt_pattern_); }

}  // namespace cinn::frontend::group_cluster
