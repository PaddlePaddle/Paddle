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

#include <memory>
#include <queue>
#include <unordered_set>
#include <vector>

#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/pattern_rewrite/drr/api/drr_pattern_context.h"
#include "paddle/ir/pattern_rewrite/drr/api/match_context.h"
#include "paddle/ir/pattern_rewrite/drr/ir_operation.h"
#include "paddle/ir/pattern_rewrite/drr/match_context_impl.h"
#include "paddle/ir/pattern_rewrite/drr/pattern_graph.h"
#include "paddle/ir/pattern_rewrite/pattern_match.h"

namespace ir {
namespace drr {

template <typename SourceOp, typename DrrFunctor>
class DrrRewritePattern : public ir::OpRewritePattern<SourceOp> {
 public:
  DrrRewritePattern(ir::IrContext* context, ir::PatternBenefit benefit)
      : ir::OpRewritePattern<SourceOp>(context, benefit) {
    DrrPatternContext drr_context;
    DrrFunctor functor;
    functor(&drr_context);

    source_pattern_graph_ = drr_context.source_pattern_graph();
    constraints_ = drr_context.constraints();
    result_pattern_graph_ = drr_context.result_pattern_graph();

    source_pattern_graph_->Print();
    result_pattern_graph_->Print();

    match_context_impl_ = std::make_unique<MatchContextImpl>();
  }

  bool Match(SourceOp op) const override {
    // Match
    auto* anchor = source_pattern_graph_->AnchorNode();
    IR_ENFORCE(anchor);
    std::unordered_set<OpCall*> drr_visited;
    std::unordered_set<Operation*> ir_visited;
    std::queue<OpCall*> drr_q;
    std::queue<ir::Operation*> ir_q;
    drr_q.push(anchor.get());
    ir_q.push(op);
    drr_visited.insert(anchor.get());
    ir_visited.insert(op);
    match_context_impl_->BindIrOperation(op->name(),
                                         std::make_shared<IrOperation>(op));
    bool Matched = true;
    size_t step = 0;
    while (!drr_q.empty()) {
      if (!Matched) break;
      IR_ENFORCE(drr_q.size() == ir_q.size());
      // if (drr_q.size() != ir_q.size()) {
      //   Matched = false;
      //   break;
      // }
      auto* drr_node = drr_q.front();
      auto* ir_node = ir_q.front();
      drr_q.pop();
      ir_q.pop();
      if (drr_node->name() != ir_node->name()) {
        Matched = false;
        break;
      }

      //
      // op's inputs
      const auto& drr_input_tensors = drr_node->inputs();
      auto ir_input_value_size = ir_node->num_operands();
      // check input's size
      if (drr_input_tensors.sizes() != ir_input_value_size) {
        Matched = false;
        break;
      }
      for (int i = 0; i < drr_input_tensors.size(); ++i) {
        if (!Matched) break;
        // check brother ops
        auto drr_brother_ops = drr_input_tensors[i]->consumers();
        auto ir_input_value = ir_node->operand(i);
        match_context_impl_->BindIrValue(drr_input_tensors[i]->name(),
                                         ir_input_value);
        if (drr_brother_ops.size() != ir_input_value.use_count()) {
          Matched = false;
          break;
        }
        for (auto* drr_brother_op : drr_brother_ops) {
          if (drr_visited.count(drr_brother_op) == 0) {
            std::pair<bool, ir::Operation*> found{false, nullptr};
            for (auto it = ir_input_value.begin(); it != ir_input_value.end();
                 ++it) {
              auto* ir_op = (*it).owner();
              if (ir_visited.count(ir_op)) {
                continue;
              }
              // todo()
              if (drr_brother_op->name() == ir_op->name()) {
                found = {true, ir_op};
                break;
              }
            }
            if (found.first) {
              drr_q.push(drr_brother_op);
              ir_q.push(found.second);
              drr_visited.insert(drr_brother_op);
              ir_visited.insert(found.second);
              match_context_impl_->BindIrOperation(
                  found.second->name(),
                  std::make_shared<IrOperation>(found.second));
            } else {
              Matched = false;
              break;
            }
          }
        }

        // check ancestor op
        auto drr_ancestor_op = drr_input_tensors[i]->producer();
        auto ir_ancestor_op = ir_input_value.GetDefiningOp();
        if (drr_ancestor_op->name() != ir_ancestor_op->name()) {
          Matched = false;
          break;
        } else {
          drr_q.push(drr_ancestor_op);
          ir_q.push(ir_ancestor_op);
          drr_visited.insert(drr_ancestor_op);
          ir_visited.insert(ir_ancestor_op);
          match_context_impl_->BindIrOperation(
              ir_ancestor_op->name(),
              std::make_shared<IrOperation>(ir_ancestor_op));
        }
      }

      //
      // op's outputs
      const auto& drr_output_tensors = drr_node->outputs();
      auto ir_output_value_size = ir_node->num_results();
      // check output's size
      if (drr_output_tensors.sizes() != ir_output_value_size) {
        Matched = false;
        break;
      }
      for (int i = 0; i < drr_output_tensors.size(); ++i) {
        if (!Matched) break;
        // check child ops
        auto drr_child_ops = drr_output_tensors[i]->consumers();
        auto ir_output_value = ir_node->result(i);
        match_context_impl_->BindIrValue(drr_output_tensors[i]->name(),
                                         ir_output_value);
        if (drr_child_ops.size() != ir_output_value.use_count()) {
          Matched = false;
          break;
        }
        for (auto* drr_child_op : drr_child_ops) {
          if (drr_visited.count(drr_child_op) == 0) {
            std::pair<bool, ir::Operation*> found{false, nullptr};
            for (auto it = ir_output_value.begin(); it != ir_output_value.end();
                 ++it) {
              auto* ir_op = (*it).owner();
              if (ir_visited.count(ir_op)) {
                continue;
              }
              // todo()
              if (drr_child_op->name() == ir_op->name()) {
                found = {true, ir_op};
                break;
              }
            }
            if (found.first) {
              drr_q.push(drr_child_op);
              ir_q.push(found.second);
              drr_visited.insert(drr_child_op);
              ir_visited.insert(found.second);
              match_context_impl_->BindIrOperation(
                  found.second->name(),
                  std::make_shared<IrOperation>(found.second));
            } else {
              Matched = false;
              break;
            }
          }
        }
      }

      step++;
    }

    if (Matched) {
      IR_ENFORCE(step == source_pattern_graph_->CountOfOpCalls());
    }
    // Matched = Matched && step == source_pattern_graph_->CountOfOpCalls();

    // Constraints
    MatchContext match_context{match_context_impl_};
    for (const auto& constraint : constraints_) {
      Matched = constraint(match_context);
      if (!Matched) break;
    }

    return Matched;
  }

  void Rewrite(SourceOp op,
               ir::PatternRewriter& rewriter) const override {  // NOLINT
    // Rewrite
  }

 private:
  std::shared_ptr<MatchContextImpl> match_context_impl_;
  std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  std::vector<Constraint> constraints_;
  std::shared_ptr<ResultPatternGraph> result_pattern_graph_;
};

}  // namespace drr
}  // namespace ir
