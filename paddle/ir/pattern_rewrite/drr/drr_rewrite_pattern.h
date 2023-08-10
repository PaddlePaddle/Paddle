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
#include "paddle/ir/pattern_rewrite/drr/ir_operation_creator.h"
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

    source_pattern_match_ctx_ = std::make_unique<MatchContextImpl>();
  }

  bool Match(SourceOp op) const override {
    // Match
    auto* anchor = source_pattern_graph_->AnchorNode();
    IR_ENFORCE(anchor);
    std::unordered_set<const OpCall*> drr_visited;
    std::unordered_set<Operation*> ir_visited;
    std::queue<const OpCall*> drr_q;
    std::queue<ir::Operation*> ir_q;
    drr_q.push(anchor);
    ir_q.push(op);
    drr_visited.insert(anchor);
    ir_visited.insert(op);
    source_pattern_match_ctx_->BindIrOperation(
        anchor, std::make_shared<IrOperation>(op));
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
      if (drr_input_tensors.size() != ir_input_value_size) {
        Matched = false;
        break;
      }
      for (size_t i = 0; i < drr_input_tensors.size(); ++i) {
        if (!Matched) break;
        // check brother ops
        auto drr_brother_ops = drr_input_tensors[i]->consumers();
        auto ir_input_value = ir_node->operand(i).source();
        source_pattern_match_ctx_->BindIrValue(
            drr_input_tensors[i]->name(),
            std::make_shared<IrValue>(ir_input_value));
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
                found = std::make_pair(true, ir_op);
                break;
              }
            }
            if (found.first) {
              drr_q.push(drr_brother_op);
              ir_q.push(found.second);
              drr_visited.insert(drr_brother_op);
              ir_visited.insert(found.second);
              source_pattern_match_ctx_->BindIrOperation(
                  drr_brother_op, std::make_shared<IrOperation>(found.second));
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
          source_pattern_match_ctx_->BindIrOperation(
              drr_ancestor_op, std::make_shared<IrOperation>(ir_ancestor_op));
        }
      }

      //
      // op's outputs
      const auto& drr_output_tensors = drr_node->outputs();
      auto ir_output_value_size = ir_node->num_results();
      // check output's size
      if (drr_output_tensors.size() != ir_output_value_size) {
        Matched = false;
        break;
      }
      for (size_t i = 0; i < drr_output_tensors.size(); ++i) {
        if (!Matched) break;
        // check child ops
        auto drr_child_ops = drr_output_tensors[i]->consumers();
        auto ir_output_value = ir_node->result(i);
        source_pattern_match_ctx_->BindIrValue(
            drr_output_tensors[i]->name(),
            std::make_shared<IrValue>(ir_output_value));
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
              source_pattern_match_ctx_->BindIrOperation(
                  drr_child_op, std::make_shared<IrOperation>(found.second));
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
    MatchContext match_context{source_pattern_match_ctx_};
    for (const auto& constraint : constraints_) {
      Matched = constraint(match_context);
      if (!Matched) break;
    }

    return Matched;
  }

  void Rewrite(SourceOp op,
               ir::PatternRewriter& rewriter) const override {  // NOLINT
    // 1. Create Operations in result_pattern_graph
    MatchContextImpl res_match_ctx = CreateOperations(
        *result_pattern_graph_, *source_pattern_match_ctx_, rewriter);

    // 2. Replace Output Values in source_pattern_graph by Output Values in
    // result_pattern_graph
    ReplaceOutputTensor(*source_pattern_match_ctx_, res_match_ctx, rewriter);

    // 3. Delete Operations in source_pattern_graph
    DeleteSourcePatternOp(*source_pattern_match_ctx_, rewriter);
  }

  MatchContextImpl CreateOperations(
      const ResultPatternGraph& result_pattern_graph,
      const MatchContextImpl& src_match_ctx,
      ir::PatternRewriter& rewriter) const {  // NOLINT
    MatchContextImpl res_match_ctx;
    // add input tensors info for res_match_ctx;
    const auto& input_tensors = result_pattern_graph.input_tensors();
    for (const auto& in_tensor : input_tensors) {
      res_match_ctx.BindIrValue(
          in_tensor,
          std::make_shared<IrValue>(src_match_ctx.GetIrValue(in_tensor)));
    }

    // topo order visit result_pattern_graph
    GraphTopo graph_topo_visit(&result_pattern_graph);
    graph_topo_visit.WalkGraphNodesTopoOrder(
        [&rewriter, &res_match_ctx](const OpCall& op_call) {
          CreateOperation(op_call, rewriter, &res_match_ctx);
        });

    return res_match_ctx;
  }

  void ReplaceOutputTensor(const MatchContextImpl& src_match_ctx,
                           const MatchContextImpl& res_match_ctx,
                           ir::PatternRewriter& rewriter) const {  // NOLINT
    for (const auto& output_name : source_pattern_graph_->output_tensors()) {
      const auto& src_ir_tensor = src_match_ctx.GetIrValue(output_name);
      const auto& res_ir_tensor = res_match_ctx.GetIrValue(output_name);
      rewriter.ReplaceAllUsesWith(src_ir_tensor.ir_value(),
                                  res_ir_tensor.ir_value());
    }
  }

  void DeleteSourcePatternOp(const MatchContextImpl& src_match_ctx,
                             ir::PatternRewriter& rewriter) const {  // NOLINT
    for (const auto& kv : src_match_ctx.operation_map()) {
      rewriter.EraseOp(kv.second->get());
    }
  }

 private:
  std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  std::vector<Constraint> constraints_;
  std::shared_ptr<ResultPatternGraph> result_pattern_graph_;

  std::shared_ptr<MatchContextImpl> source_pattern_match_ctx_;
};

}  // namespace drr
}  // namespace ir
