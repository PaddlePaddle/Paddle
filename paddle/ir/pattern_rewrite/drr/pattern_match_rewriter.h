// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/ir/pattern_rewrite/drr/source_pattern_graph.h"

namespace cinn {
namespace hlir {
namespace drr {

class PatternMatchRewriter {
 public:
  PatternMatchRewriter(const SourcePatternGraph* source_pattern_graph,
                       const std::vector<Constrain*>& constraints,
                       const ResultPatternGraph* result_pattern_graph)
      : source_pattern_graph_(source_pattern_graph),
        constraints_(constraints),
        result_pattern_graph_(result_pattern_graph) {}

  bool Apply(ir::Program* program) {
    // step 1
    auto result = MatchPattern(source_pattern_graph_, program);
    // step 2
    if (ConstrainsCheck(result, constraints_)) {
      // step 3
      RewritePattern(result_pattern_graph_, result);
      return true;
    }
    return false;
  }

 private:
  const SourcePatternGraph* source_pattern_graph_;
  const std::vector<Constrain*>& constraints_;
  const ResultPatternGraph* result_pattern_graph_;
};

class DrrDagRewritePattern : public ir::RewritePattern {
 public:
  DrrDagRewritePattern(const SourcePatternGraph* source_pattern_graph,
                       const std::vector<Constrain*>& constraints,
                       const ResultPatternGraph* result_pattern_graph)
      : source_pattern_graph(source_pattern_graph),
        constraints(constraints),
        result_pattern_graph(result_pattern_graph) {}

  bool MatchAndRewrite(paddle::dialect::TransposeOp op,
                       ir::PatternRewriter& rewriter) const override {
    // Match
    auto sink = source_pattern_graph_->SinkNode();
    std::unordered_map<const OpCall*, const ir::Operation*> op_map;
    std::unordered_map<const Tensor*, const ir::Value*> tensor_map;
    auto bfs = [](const OpCall* sink, const ir::Operation* op) {
      auto is_op_matched =
          [&](const OpCall* op_call, const ir::Operation* op) {
            return op_call->name() == op->name();
          } std::add_cv_t<const OpCall*>
              op_queue;
      while (!op_queue.empty()) {
        auto cur_op_call = op_queue.front();
        op_queue.pop();
        if (is_op_matched(cur_op_call, op_map[cur_op_call])) {
          VisitInput(cur_op_call, op_map[cur_op_call]);
          VisitOutput(cur_op_call, op_map[cur_op_call]);
        } else {
          return false;
        }
      }
      return true;
    };
    bfs();

    // Constrains
    MatchContext match_context = BuildMatchContext(op_map, tensor_map);
    for (auto constrain : constraints_) {
      if (!constrain(match_context)) {
        return false;
      }
    }

    // Rewrite
    // 通过拓扑排序从输入节点开始逐个创建，但从字符串到ir
    // Op类的创建只能通过swich-case的方式，这里可能需要使用代码生成来降低成本
    CreateOp(result_pattern_graph_, rewriter);
    ReplaceOutput(source_pattern_graph_, result_pattern_graph_, rewriter);
    // source_pattern 里的全部 OP 都需要删除
    DeleteOriginOp(source_pattern_graph_, rewriter);

    return true;
  }

  ir::Operation* CreateOp(const OpCall& op_call) {
    if (op_call.name() == "Transpose") {
      return new paddle::dialect::TransposeOp();
    } else if (op_call.name() == "TransposeNCHW") {
    }
  }

 private:
  const SourcePatternGraph* source_pattern_graph_;
  const std::vector<Constrain*>& constraints_;
  const ResultPatternGraph* result_pattern_graph_;
};

}  // namespace drr
}  // namespace hlir
}  // namespace cinn
