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

PatternNode::PatternNode(pir::Operation* op)
    : sink_op_(op), stmt_pattern_(ConvertToStmtPattern(op)) {}

PatternNode::PatternNode(PatternNodePtr fused_up_node,
                         PatternNodePtr fused_down_node)
    : sink_op_(fused_down_node->sink_op_),
      stmt_pattern_(MergePattern(fused_up_node->stmt_pattern_,
                                 fused_down_node->stmt_pattern_)) {}

std::vector<pir::Operation*> PatternNode::GetOps() const {
  return GetOpsInPattern(stmt_pattern_);
}

bool PatternNode::IsTrivial() const { return IsTrivialPattern(stmt_pattern_); }
bool PatternNode::IsReduce() const { return IsReducePattern(stmt_pattern_); }
bool PatternNode::IsReduceTree() const {
  return IsReduceTreePattern(stmt_pattern_);
}
bool PatternNode::IsUnsupport() const {
  return IsUnsupportPattern(stmt_pattern_);
}
bool PatternNode::IsReduceTrivial() const {
  return IsReduceTrivialPattern(stmt_pattern_);
}
std::string PatternNode::DebugStr() const {
  std::stringstream ss;
  ss << "Node: " << this << ", Pattern: " << GetPatternName(stmt_pattern_)
     << "\n    -u>:  ";
  for (const auto& u : upstream_) {
    ss << u << ", ";
  }
  ss << "\n    <d-:  ";
  for (const auto& d : downstream_) {
    ss << d << ", ";
  }
  return ss.str();
}

}  // namespace cinn::frontend::group_cluster
