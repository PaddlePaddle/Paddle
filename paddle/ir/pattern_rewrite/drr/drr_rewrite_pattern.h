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
  explicit DrrRewritePattern(ir::IrContext* context,
                             ir::PatternBenefit benefit = 1)
      : ir::OpRewritePattern<SourceOp>(context, benefit) {
    DrrPatternContext drr_context;
    DrrFunctor functor;
    functor(&drr_context);

    source_pattern_graph_ = drr_context.source_pattern_graph();
    constraints_ = drr_context.constraints();
    result_pattern_graph_ = drr_context.result_pattern_graph();

    source_pattern_graph_->Print();
    result_pattern_graph_->Print();

    VLOG(1) << "#### source_pattern_graph_: " << source_pattern_graph_.get()
            << " : " << source_pattern_graph_.use_count();
  }

  bool MatchAndRewrite(SourceOp op,
                       PatternRewriter& rewriter) const override {  // NOLINT
    std::shared_ptr<MatchContextImpl> src_match_ctx =
        std::make_shared<MatchContextImpl>();
    if (match(op, src_match_ctx)) {
      Rewrite(op, *src_match_ctx, rewriter);
      return true;
    }
    return false;
  }

  bool match(SourceOp op,
             std::shared_ptr<MatchContextImpl> source_pattern_match_ctx) const {
    // Match
    VLOG(1) << "######### DrrRewritePattern Match ";
    VLOG(1) << "#### source_pattern_graph_: " << source_pattern_graph_.get()
            << " : " << source_pattern_graph_.use_count();

    auto* anchor = source_pattern_graph_->AnchorNode();
    IR_ENFORCE(anchor);
    std::unordered_set<const OpCall*> drr_visited;
    std::unordered_set<const Operation*> ir_visited;
    std::queue<const OpCall*> drr_q;
    std::queue<const ir::Operation*> ir_q;
    drr_q.push(anchor);
    ir_q.push(op);
    drr_visited.insert(anchor);
    ir_visited.insert(op);
    source_pattern_match_ctx->BindIrOperation(
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
      VLOG(1) << "#####  drr_node->name() " << drr_node->name();
      VLOG(1) << "#####  ir_node->name() " << ir_node->name() << " @"
              << ir_node;
      if (drr_node->name() != ir_node->name()) {
        Matched = false;
        break;
      }

      // op's inputs
      const auto& drr_input_tensors = drr_node->inputs();
      auto ir_input_value_size = ir_node->num_operands();
      // check input's size
      if (drr_input_tensors.size() != ir_input_value_size) {
        Matched = false;
        break;
      }
      VLOG(1) << "###### drr_input_tensors " << drr_input_tensors.size();
      for (size_t i = 0; i < drr_input_tensors.size(); ++i) {
        VLOG(1) << "###### " << i << " " << drr_input_tensors[i]->name();
        if (!Matched) break;
        // check brother ops
        auto drr_brother_ops = drr_input_tensors[i]->consumers();
        VLOG(1) << "###### ir_node->operand(i).source()";
        auto ir_input_value = ir_node->operand(i).source();
        VLOG(1) << "###### BindIrValue";

        source_pattern_match_ctx->BindIrValue(
            drr_input_tensors[i]->name(),
            std::make_shared<IrValue>(ir_input_value));
        if (drr_brother_ops.size() != ir_input_value.use_count()) {
          VLOG(1) << "###### Matched = false";
          Matched = false;
          break;
        }
        VLOG(1) << "###### drr_brother_ops " << drr_brother_ops.size();

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
              source_pattern_match_ctx->BindIrOperation(
                  drr_brother_op, std::make_shared<IrOperation>(found.second));
            } else {
              Matched = false;
              break;
            }
          }
        }

        VLOG(1) << "######  check ancestor op";
        VLOG(1) << "###### " << drr_input_tensors[i]->name() << " i: " << i;

        if (source_pattern_graph_->input_tensors().count(
                drr_input_tensors[i]->name())) {
          continue;
        }

        // check ancestor op
        auto drr_ancestor_op = drr_input_tensors[i]->producer();
        VLOG(1) << "##### drr_ancestor_op->name(): " << drr_ancestor_op->name();

        auto ir_ancestor_op = ir_input_value.GetDefiningOp();
        VLOG(1) << "##### ir_ancestor_op->name(): " << ir_ancestor_op->name();
        if (drr_ancestor_op->name() != ir_ancestor_op->name()) {
          Matched = false;
          break;
        } else {
          drr_q.push(drr_ancestor_op);
          ir_q.push(ir_ancestor_op);
          drr_visited.insert(drr_ancestor_op);
          ir_visited.insert(ir_ancestor_op);
          source_pattern_match_ctx->BindIrOperation(
              drr_ancestor_op, std::make_shared<IrOperation>(ir_ancestor_op));
        }
      }

      VLOG(1) << "######  drr_node->outputs()";

      // op's outputs
      const auto& drr_output_tensors = drr_node->outputs();
      auto ir_output_value_size = ir_node->num_results();
      // check output's size
      if (drr_output_tensors.size() != ir_output_value_size) {
        Matched = false;
        break;
      }
      VLOG(1) << "######  ir_output_value_size: " << ir_output_value_size;
      for (size_t i = 0; i < drr_output_tensors.size(); ++i) {
        VLOG(1) << "###### drr_output_tensors[ i ]: " << i << " "
                << drr_output_tensors[i]->name();
        if (!Matched) break;
        // check child ops
        auto drr_child_ops = drr_output_tensors[i]->consumers();
        auto ir_output_value = ir_node->result(i);
        source_pattern_match_ctx->BindIrValue(
            drr_output_tensors[i]->name(),
            std::make_shared<IrValue>(ir_output_value));
        if (source_pattern_graph_->output_tensors().count(
                drr_output_tensors[i]->name())) {
          continue;
        }
        if (drr_child_ops.size() != ir_output_value.use_count()) {
          Matched = false;
          break;
        }
        VLOG(1) << "###### drr_child_ops: " << drr_child_ops.size();
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
              source_pattern_match_ctx->BindIrOperation(
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
    } else {
      return Matched;
    }
    // Matched = Matched && step == source_pattern_graph_->CountOfOpCalls();

    // Constraints
    MatchContext match_context{source_pattern_match_ctx};
    for (const auto& constraint : constraints_) {
      Matched = constraint(match_context);
      if (!Matched) break;
    }

    VLOG(1) << "######### DrrRewritePattern Match success";

    return Matched;
  }

  void Rewrite(SourceOp op,
               const MatchContextImpl& source_pattern_match_ctx,
               ir::PatternRewriter& rewriter) const {  // NOLINT
    VLOG(1) << "######### DrrRewritePattern Rewrite";
    // 1. Create Operations in result_pattern_graph
    MatchContextImpl res_match_ctx = CreateOperations(
        *result_pattern_graph_, source_pattern_match_ctx, rewriter);

    VLOG(1) << "######### DrrRewritePattern ReplaceOutputTensor";
    // 2. Replace Output Values in source_pattern_graph by Output Values in
    // result_pattern_graph
    ReplaceOutputTensor(source_pattern_match_ctx, res_match_ctx, rewriter);

    VLOG(1) << "######### DrrRewritePattern DeleteSourcePatternOp";
    // 3. Delete Operations in source_pattern_graph
    DeleteSourcePatternOp(
        *source_pattern_graph_, source_pattern_match_ctx, rewriter);
  }

 private:
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
      VLOG(1) << "######## ReplaceOutputTensor " << output_name;
      if (result_pattern_graph_->output_tensors().count(output_name)) {
        VLOG(1) << "######## GetIrValue src_ir_tensor ";
        const auto& src_ir_tensor = src_match_ctx.GetIrValue(output_name);
        VLOG(1) << "######## GetIrValue res_ir_tensor";
        const auto& res_ir_tensor = res_match_ctx.GetIrValue(output_name);
        VLOG(1) << "##### Before src_ir_tensor " << src_ir_tensor.get().impl()
                << " use_count(): " << src_ir_tensor.get().use_count() << " "
                << res_ir_tensor.get().impl()
                << " res_ir_tensor.get(): " << res_ir_tensor.get().use_count();
        rewriter.ReplaceAllUsesWith(src_ir_tensor.get(), res_ir_tensor.get());
        VLOG(1) << "##### After src_ir_tensor use_count(): "
                << src_ir_tensor.get().use_count()
                << " res_ir_tensor.get(): " << res_ir_tensor.get().use_count();
      } else {
        LOG(WARNING) << "The output tensor (" << output_name
                     << ") in the source_pattern_graph is not the output "
                        "tensor in result_pattern_graph.";
      }
    }
  }

  void DeleteSourcePatternOp(const SourcePatternGraph& source_pattern_graph,
                             const MatchContextImpl& src_match_ctx,
                             ir::PatternRewriter& rewriter) const {  // NOLINT
    std::vector<const OpCall*> topo_order_ops;
    GraphTopo graph_topo_visit(&source_pattern_graph);
    graph_topo_visit.WalkGraphNodesTopoOrder(
        [&topo_order_ops](const OpCall& op_call) {
          topo_order_ops.push_back(&op_call);
        });
    // Delete Operation with topo order from output tensors.
    std::for_each(
        topo_order_ops.rbegin(),
        topo_order_ops.rend(),
        [&src_match_ctx, &rewriter](const OpCall* op_call) {
          auto* op = src_match_ctx.operation_map().at(op_call)->get();
          VLOG(1) << "Delete (" << op_call->name() << " @" << op_call << " :@"
                  << op << ") in source_pattern_graph ";
          for (const auto& result : op->results()) {
            VLOG(1) << "###### " << result.value_impl()
                    << " use_count : " << result.use_count();
          }
          for (const auto& output : op_call->outputs()) {
            auto value = src_match_ctx.GetIrValue(output->name()).get();
            VLOG(1) << "###### " << output->name() << " " << value.impl()
                    << " use_count : " << value.use_count();
          }
          rewriter.EraseOp(src_match_ctx.operation_map().at(op_call)->get());
        });
  }

 private:
  std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  std::vector<Constraint> constraints_;
  std::shared_ptr<ResultPatternGraph> result_pattern_graph_;
};

}  // namespace drr
}  // namespace ir
