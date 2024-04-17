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

#include "paddle/cinn/operator_fusion/pattern.h"
#include "paddle/cinn/operator_fusion/pattern_fuser.h"
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

template <typename T>
struct PatternNode {
  using PatternNodePtr = std::shared_ptr<PatternNode<T>>;

  explicit PatternNode(const PatternContent<T>& content)
      : sink_op_(content.op), stmt_pattern_(ConvertToStmtPattern<T>(content)) {}

  explicit PatternNode(PatternNodePtr fused_up_node,
                       PatternNodePtr fused_down_node)
      : sink_op_(fused_down_node->sink_op_),
        stmt_pattern_(MergePattern<T>(fused_up_node->stmt_pattern_,
                                      fused_down_node->stmt_pattern_)) {
    // Update the upstream & downstream
    ExtendVector(&upstream_, fused_up_node->upstream());
    ExtendVector(&upstream_, fused_down_node->upstream());
    RemoveFromVector(&upstream_, fused_up_node);

    ExtendVector(&downstream_, fused_up_node->downstream());
    ExtendVector(&downstream_, fused_down_node->downstream());
    RemoveFromVector(&downstream_, fused_down_node);
  }

  std::string DebugStr() const {
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

  pir::Operation* sink_op() const { return sink_op_; }
  const StmtPattern<T>& stmt_pattern() const { return stmt_pattern_; }
  void set_stmt_pattern(const StmtPattern<T>& pattern) {
    stmt_pattern_ = pattern;
  }
  const std::vector<PatternNodePtr>& upstream() const { return upstream_; }
  const std::vector<PatternNodePtr>& downstream() const { return downstream_; }
  void AddNodeToUpstream(PatternNodePtr node) { upstream_.push_back(node); }
  void AddNodeToDownstream(PatternNodePtr node) { downstream_.push_back(node); }
  void RemoveNodeFromUpstream(PatternNodePtr node) {
    RemoveFromVector(&upstream_, node);
  }
  void RemoveNodeFromDownstream(PatternNodePtr node) {
    RemoveFromVector(&downstream_, node);
  }
  void ClearUpstream() { upstream_.clear(); }
  void ClearDownstream() { downstream_.clear(); }

 private:
  StmtPattern<T> stmt_pattern_;
  pir::Operation* sink_op_;

  std::vector<PatternNodePtr> upstream_;
  std::vector<PatternNodePtr> downstream_;
};

template <typename T>
using PatternNodePtr = std::shared_ptr<PatternNode<T>>;
}  // namespace cinn::fusion
