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

PatternNode::PatternNode(PatternNode* fused_up_node,
                         PatternNode* fused_down_node)
    : stmt_pattern_(MergePattern(fused_up_node->stmt_pattern_,
                                 fused_down_node->stmt_pattern_)) {
  sink_op_ = fused_down_node->sink_op_;

  upstream_.insert(fused_up_node->upstream_.begin(),
                   fused_up_node->upstream_.end());
  upstream_.insert(fused_down_node->upstream_.begin(),
                   fused_down_node->upstream_.end());
  upstream_.erase(fused_up_node);

  downstream_.insert(fused_up_node->downstream_.begin(),
                     fused_up_node->downstream_.end());
  downstream_.insert(fused_down_node->downstream_.begin(),
                     fused_down_node->downstream_.end());
  downstream_.erase(fused_down_node);

  for (const auto& upstream_node : upstream_) {
    if (upstream_node->downstream_.find(fused_up_node) !=
        upstream_node->downstream_.end()) {
      upstream_node->downstream_.erase(fused_up_node);
    }
    if (upstream_node->downstream_.find(fused_down_node) !=
        upstream_node->downstream_.end()) {
      upstream_node->downstream_.erase(fused_down_node);
    }
  }

  for (const auto& downstream_node : downstream_) {
    if (downstream_node->upstream_.find(fused_up_node) !=
        downstream_node->upstream_.end()) {
      downstream_node->upstream_.erase(fused_up_node);
    }
    if (downstream_node->upstream_.find(fused_down_node) !=
        downstream_node->upstream_.end()) {
      downstream_node->upstream_.erase(fused_down_node);
    }
  }
}

std::unordered_set<const pir::Operation*> PatternNode::GetOps() const {
  return GetOpsInPattern(stmt_pattern_);
}

bool PatternNode::IsTrivial() const { return IsTrivialPattern(stmt_pattern_); }

}  // namespace cinn::frontend::group_cluster
