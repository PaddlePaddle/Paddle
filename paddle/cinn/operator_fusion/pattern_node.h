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
#include "paddle/cinn/operator_fusion/pir_graph_analyzing/shardable_axes_base.h"
#include "paddle/cinn/operator_fusion/utils.h"

namespace cinn::fusion {

struct PatternNode {
  using PatternNodePtr = std::shared_ptr<PatternNode>;
  using MergePatternFn =
      std::function<StmtPattern(const StmtPattern&, const StmtPattern&)>;

  explicit PatternNode(const PatternContent& content,
                       const ShardableAxesSignature& axes)
      : sink_op_(content.op),
        stmt_pattern_(ConvertToStmtPattern(content)),
        axes_(axes) {}

  explicit PatternNode(PatternNodePtr fused_up_node,
                       PatternNodePtr fused_down_node,
                       MergePatternFn merge_pattern_fn)
      : sink_op_(fused_down_node->sink_op_),
        stmt_pattern_(merge_pattern_fn(fused_up_node->stmt_pattern_,
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
    ss << "Node: " << this << ", Pattern: " << GetPatternName(stmt_pattern())
       << ", ID: " << GetPatternId(stmt_pattern());
    ss << "\n    -u>:  ";
    for (const auto& u : upstream_) {
      ss << GetPatternId(u->stmt_pattern()) << "(" << u << "), ";
    }
    ss << "\n    <d-:  ";
    for (const auto& d : downstream_) {
      ss << GetPatternId(d->stmt_pattern()) << "(" << d << "), ";
    }
    ss << "\n" << axes_.DebugStr();
    pir::IrPrinter printer(ss);
    if (GetPatternName(stmt_pattern_) == AnchorPattern::name()) {
      ss << "\n anchor: ";
      auto anchor_op =
          std::get<AnchorPattern>(stmt_pattern_).anchor().defining_op();
      printer.PrintOperation(const_cast<pir::Operation*>(anchor_op));
    }
    ss << "\nOps in pattern:" << std::endl;
    ss << OpsDebugStr(GetOpsInPattern(this->stmt_pattern()));
    return ss.str();
  }

  pir::Operation* sink_op() const { return sink_op_; }
  const StmtPattern& stmt_pattern() const { return stmt_pattern_; }
  void set_stmt_pattern(const StmtPattern& pattern) { stmt_pattern_ = pattern; }
  const std::vector<PatternNodePtr>& upstream() const { return upstream_; }
  const std::vector<PatternNodePtr>& downstream() const { return downstream_; }
  std::string name() const { return GetPatternName(stmt_pattern_); }
  std::string id() const { return GetPatternId(stmt_pattern_); }
  void set_return() const { SetReturnInstr(stmt_pattern_); }
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
  void UniqueUpstream() { upstream_ = UniqueVectorBySet(upstream_); }
  void UniqueDownstream() { downstream_ = UniqueVectorBySet(downstream_); }
  void AppendInstr(FusionInstrPtr instr) {
    GetFusionTracker(stmt_pattern_)->append(instr);
  }
  void UpdateTracker() { PatternUpdateTracker(stmt_pattern_); }

 private:
  StmtPattern stmt_pattern_;
  pir::Operation* sink_op_;

  std::vector<PatternNodePtr> upstream_;
  std::vector<PatternNodePtr> downstream_;

  ShardableAxesSignature axes_;
};

using PatternNodePtr = std::shared_ptr<PatternNode>;
}  // namespace cinn::fusion
