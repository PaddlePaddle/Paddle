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

#include "paddle/fluid/ir/drr/api/drr_pattern_context.h"
#include "paddle/fluid/ir/drr/api/match_context.h"
#include "paddle/fluid/ir/drr/ir_operation.h"
#include "paddle/fluid/ir/drr/ir_operation_creator.h"
#include "paddle/fluid/ir/drr/match_context_impl.h"
#include "paddle/fluid/ir/drr/pattern_graph.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/pattern_rewrite/pattern_match.h"

namespace ir {
namespace drr {

template <typename DrrPattern>
class DrrRewritePattern : public ir::RewritePattern {
 public:
  explicit DrrRewritePattern(const DrrPatternContext& drr_context,
                             ir::IrContext* context,
                             ir::PatternBenefit benefit = 1)
      : ir::RewritePattern(
            drr_context.source_pattern_graph()->AnchorNode()->name(),
            benefit,
            context,
            {}),
        source_pattern_graph_(drr_context.source_pattern_graph()),
        constraints_(drr_context.constraints()),
        result_pattern_graph_(drr_context.result_pattern_graph()) {
    IR_ENFORCE(source_pattern_graph_->owned_op_call().size(),
               "source_pattern_graph is empty, please check the drr pattern "
               "define code.");
    source_pattern_graph_->Print();
    result_pattern_graph_->Print();
  }

  bool MatchAndRewrite(ir::Operation* op,
                       PatternRewriter& rewriter) const override {  // NOLINT
    std::shared_ptr<MatchContextImpl> src_match_ctx =
        std::make_shared<MatchContextImpl>();
    if (PatternGraphMatch(op, src_match_ctx)) {
      VLOG(4) << "DRR pattern (" << ir::get_type_name<DrrPattern>()
              << ") is matched in program.";
      PatternGraphRewrite(*src_match_ctx, rewriter);
      return true;
    }
    return false;
  }

 private:
  bool PatternGraphMatch(
      ir::Operation* op,
      const std::shared_ptr<MatchContextImpl>& source_pattern_match_ctx) const {
    // Match
    VLOG(1) << "PatternGraphMatch Start: op(" << op->name() << ")";
    const auto* anchor = source_pattern_graph_->AnchorNode();
    IR_ENFORCE(anchor);
    std::unordered_set<const OpCall*> drr_visited;
    std::unordered_set<Operation*> ir_visited;
    std::queue<const OpCall*> drr_q;
    std::queue<ir::Operation*> ir_q;
    drr_q.push(anchor);
    ir_q.push(op);
    drr_visited.insert(anchor);
    ir_visited.insert(op);
    bool matched = true;
    size_t step = 0;
    while (!drr_q.empty()) {
      if (!matched) break;

      IR_ENFORCE(drr_q.size() == ir_q.size());

      auto* drr_node = drr_q.front();
      auto* ir_node = ir_q.front();
      drr_q.pop();
      ir_q.pop();
      VLOG(1) << "####### drr_node->name(): " << drr_node->name()
              << " ir_node->name(): " << ir_node->name();
      if (drr_node->name() != ir_node->name()) {
        matched = false;
        break;
      }
      source_pattern_match_ctx->BindIrOperation(
          drr_node, std::make_shared<IrOperation>(ir_node));

      // op's inputs
      const auto& drr_input_tensors = drr_node->inputs();
      auto ir_input_value_size = ir_node->num_operands();
      // check input's size
      if (drr_input_tensors.size() != ir_input_value_size) {
        matched = false;
        break;
      }
      for (size_t i = 0; i < drr_input_tensors.size(); ++i) {
        if (!matched) break;
        // check brother ops
        const auto& drr_brother_ops = drr_input_tensors[i]->consumers();
        auto ir_input_value = ir_node->operand(i).source();

        source_pattern_match_ctx->BindIrValue(
            drr_input_tensors[i]->name(),
            std::make_shared<IrValue>(ir_input_value));
        VLOG(1) << "###### " << i << " " << drr_brother_ops.size() << " "
                << ir_input_value.use_count();
        if (drr_brother_ops.size() != ir_input_value.use_count()) {
          matched = false;
          break;
        }

        for (auto* drr_brother_op : drr_brother_ops) {
          if (drr_visited.count(drr_brother_op)) {
            continue;
          }
          std::pair<bool, ir::Operation*> found{false, nullptr};
          for (auto& it : ir_input_value) {
            auto* ir_op = it.owner();
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
          } else {
            matched = false;
            break;
          }
        }

        if (source_pattern_graph_->input_tensors().count(
                drr_input_tensors[i]->name())) {
          continue;
        }

        VLOG(1) << "###### drr_ancestor_op";

        // check ancestor op
        auto* drr_ancestor_op = drr_input_tensors[i]->producer();
        auto* ir_ancestor_op = ir_input_value.GetDefiningOp();
        if (drr_ancestor_op->name() != ir_ancestor_op->name()) {
          matched = false;
          break;
        }

        if (drr_visited.count(drr_ancestor_op) == 0) {
          drr_q.push(drr_ancestor_op);
          ir_q.push(ir_ancestor_op);
          drr_visited.insert(drr_ancestor_op);
          ir_visited.insert(ir_ancestor_op);
        }
      }

      VLOG(1) << "###### drr_output_tensors";

      // op's outputs
      const auto& drr_output_tensors = drr_node->outputs();
      auto ir_output_value_size = ir_node->num_results();
      // check output's size
      if (drr_output_tensors.size() != ir_output_value_size) {
        matched = false;
        break;
      }

      for (size_t i = 0; i < drr_output_tensors.size(); ++i) {
        if (!matched) break;
        // check child ops
        const auto& drr_child_ops = drr_output_tensors[i]->consumers();
        auto ir_output_value = ir_node->result(i);
        source_pattern_match_ctx->BindIrValue(
            drr_output_tensors[i]->name(),
            std::make_shared<IrValue>(ir_output_value));
        if (source_pattern_graph_->output_tensors().count(
                drr_output_tensors[i]->name())) {
          continue;
        }
        if (drr_child_ops.size() != ir_output_value.use_count()) {
          matched = false;
          break;
        }

        for (auto* drr_child_op : drr_child_ops) {
          if (drr_visited.count(drr_child_op)) {
            continue;
          }
          std::pair<bool, ir::Operation*> found{false, nullptr};
          for (auto& it : ir_output_value) {
            auto* ir_op = it.owner();
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
          } else {
            matched = false;
            break;
          }
        }
      }

      step++;
    }

    if (matched) {
      VLOG(1) << "##### step: " << step
              << "  ### source_pattern_graph_->CountOfOpCalls(): "
              << source_pattern_graph_->CountOfOpCalls();
      IR_ENFORCE(step == source_pattern_graph_->CountOfOpCalls());
    } else {
      return matched;
    }
    // matched = matched && step == source_pattern_graph_->CountOfOpCalls();

    // Constraints
    MatchContext match_context{source_pattern_match_ctx};
    for (const auto& constraint : constraints_) {
      matched = constraint(match_context);
      if (!matched) break;
    }

    return matched;
  }

  void PatternGraphRewrite(const MatchContextImpl& source_pattern_match_ctx,
                           ir::PatternRewriter& rewriter) const {  // NOLINT
    // 1. Create Operations in result_pattern_graph
    MatchContextImpl res_match_ctx = CreateOperations(
        *result_pattern_graph_, source_pattern_match_ctx, rewriter);

    VLOG(1) << "#### CreateOperations";

    // 2. Process Assign Tensor
    RebindIrTensorForAssignTensor(*result_pattern_graph_, &res_match_ctx);

    VLOG(1) << "#### RebindIrTensorForAssignTensor";

    // 3. Replace Output Values in source_pattern_graph by Output Values in
    // result_pattern_graph
    ReplaceOutputTensor(source_pattern_match_ctx, res_match_ctx, rewriter);

    VLOG(1) << "#### ReplaceOutputTensor";

    // 4. Delete Operations in source_pattern_graph
    DeleteSourcePatternOp(
        *source_pattern_graph_, source_pattern_match_ctx, rewriter);
  }

 private:
  MatchContextImpl CreateOperations(
      const ResultPatternGraph& result_pattern_graph,
      const MatchContextImpl& src_match_ctx,
      ir::PatternRewriter& rewriter) const {  // NOLINT
    MatchContextImpl res_match_ctx;
    // add input tensors info for res_match_ctx
    for (const auto& in_tensor : result_pattern_graph.input_tensors()) {
      res_match_ctx.BindIrValue(
          in_tensor,
          std::make_shared<IrValue>(src_match_ctx.GetIrValue(in_tensor)));
    }
    VLOG(1) << "#### WalkGraphNodesTopoOrder";
    // topo order visit result_pattern_graph
    GraphTopo graph_topo_visit(&result_pattern_graph);
    graph_topo_visit.WalkGraphNodesTopoOrder(
        [&src_match_ctx, &rewriter, &res_match_ctx](const OpCall& op_call) {
          CreateOperation(op_call, src_match_ctx, rewriter, &res_match_ctx);
        });

    return res_match_ctx;
  }

  void RebindIrTensorForAssignTensor(
      const ResultPatternGraph& result_pattern_graph,
      MatchContextImpl* res_match_ctx) const {
    const auto& tensor_assign_map = result_pattern_graph.tensor_assign_map();
    for (const auto& kv : tensor_assign_map) {
      const auto& src_tensor_name = kv.first;
      const auto& dst_tensor_name = kv.second;
      res_match_ctx->BindIrValue(
          src_tensor_name,
          std::make_shared<IrValue>(
              res_match_ctx->GetIrValue(dst_tensor_name)));
    }
  }

  void ReplaceOutputTensor(const MatchContextImpl& src_match_ctx,
                           const MatchContextImpl& res_match_ctx,
                           ir::PatternRewriter& rewriter) const {  // NOLINT
    for (const auto& output_name : source_pattern_graph_->output_tensors()) {
      if (result_pattern_graph_->output_tensors().count(output_name)) {
        const auto& src_ir_tensor = src_match_ctx.GetIrValue(output_name);
        const auto& res_ir_tensor = res_match_ctx.GetIrValue(output_name);
        rewriter.ReplaceAllUsesWith(src_ir_tensor.get(), res_ir_tensor.get());
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
          VLOG(6) << "Delete (" << op_call->name() << " @" << op_call << " :@"
                  << op << ") in source_pattern_graph ";
          rewriter.EraseOp(src_match_ctx.operation_map().at(op_call)->get());
        });
  }

  const std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  const std::vector<Constraint> constraints_;
  const std::shared_ptr<ResultPatternGraph> result_pattern_graph_;
};

}  // namespace drr
}  // namespace ir
