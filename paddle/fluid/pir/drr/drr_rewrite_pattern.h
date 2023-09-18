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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/pir/drr/api/drr_pattern_context.h"
#include "paddle/fluid/pir/drr/api/match_context.h"
#include "paddle/fluid/pir/drr/ir_operation.h"
#include "paddle/fluid/pir/drr/ir_operation_creator.h"
#include "paddle/fluid/pir/drr/match_context_impl.h"
#include "paddle/fluid/pir/drr/pattern_graph.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/type_name.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"

namespace pir {
namespace drr {

template <typename DrrPattern>
class DrrRewritePattern : public pir::RewritePattern {
 public:
  explicit DrrRewritePattern(const DrrPatternContext& drr_context,
                             pir::IrContext* context,
                             pir::PatternBenefit benefit = 1)
      : pir::RewritePattern(
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

  bool MatchAndRewrite(pir::Operation* op,
                       PatternRewriter& rewriter) const override {  // NOLINT
    std::shared_ptr<MatchContextImpl> src_match_ctx =
        std::make_shared<MatchContextImpl>();
    if (PatternGraphMatchV2(op, src_match_ctx.get())) {
      VLOG(4) << "DRR pattern (" << pir::get_type_name<DrrPattern>()
              << ") is matched in program.";
      PatternGraphRewrite(*src_match_ctx, rewriter);
      return true;
    }
    return false;
  }

 private:
  bool PatternGraphMatch(
      pir::Operation* op,
      const std::shared_ptr<MatchContextImpl>& source_pattern_match_ctx) const {
    // Match
    VLOG(6) << "PatternGraphMatch Start: op(" << op->name() << ")";
    const auto* anchor = source_pattern_graph_->AnchorNode();
    IR_ENFORCE(anchor);
    std::unordered_set<const OpCall*> drr_visited;
    std::unordered_set<pir::Operation*> ir_visited;
    std::queue<const OpCall*> drr_q;
    std::queue<pir::Operation*> ir_q;
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
      if (drr_node->name() != ir_node->name()) {
        matched = false;
        VLOG(6) << " --- match false: " << drr_node->name();
        break;
      } else {
        VLOG(6) << " --- match true: " << drr_node->name();
      }
      source_pattern_match_ctx->BindIrOperation(
          drr_node, std::make_shared<IrOperation>(ir_node));

      // op's inputs
      const auto& drr_input_tensors = drr_node->inputs();
      auto ir_input_value_size = ir_node->num_operands();
      // check input's size
      if (drr_input_tensors.size() != ir_input_value_size) {
        matched = false;
        VLOG(6) << " --- match false: " << drr_input_tensors.size()
                << " not equal " << ir_input_value_size;
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
        // Input tensor is optional(or none)
        if (!ir_input_value) {
          if (drr_brother_ops.size() != 1) {  // Only used by current op
            matched = false;
            VLOG(6) << " --- match false: drr_brother_ops is "
                    << drr_brother_ops.size()
                    << ", but ir_input_value is null ";
          }
          continue;
        }
        if (drr_brother_ops.size() != ir_input_value.use_count()) {
          matched = false;
          VLOG(6) << " --- match false: " << drr_brother_ops.size()
                  << " not equal " << ir_input_value.use_count();
          break;
        }

        for (auto* drr_brother_op : drr_brother_ops) {
          if (drr_visited.count(drr_brother_op)) {
            continue;
          }
          std::pair<bool, pir::Operation*> found{false, nullptr};
          for (auto it = ir_input_value.use_begin();
               it != ir_input_value.use_end();
               ++it) {
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
          }
          //  else {
          //   VLOG(6) << " --- match false: brother op not same";
          //   matched = false;
          //   break;
          // }
        }

        if (source_pattern_graph_->input_tensors().count(
                drr_input_tensors[i]->name())) {
          continue;
        }

        // check ancestor op
        auto* drr_ancestor_op = drr_input_tensors[i]->producer();
        auto* ir_ancestor_op = ir_input_value.GetDefiningOp();
        if (drr_ancestor_op->name() != ir_ancestor_op->name()) {
          VLOG(6) << " --- match false: ancestor op not same";
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

      // op's outputs
      const auto& drr_output_tensors = drr_node->outputs();
      auto ir_output_value_size = ir_node->num_results();
      // check output's size
      if (drr_output_tensors.size() != ir_output_value_size) {
        matched = false;
        VLOG(6) << " --- match false: " << drr_output_tensors.size()
                << " not equal " << ir_output_value_size;
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
          VLOG(6) << " --- match false: " << drr_child_ops.size()
                  << " not equal " << ir_output_value.use_count();
          break;
        }

        for (auto* drr_child_op : drr_child_ops) {
          if (drr_visited.count(drr_child_op)) {
            continue;
          }
          std::pair<bool, pir::Operation*> found{false, nullptr};
          for (auto it = ir_output_value.use_begin();
               it != ir_output_value.use_end();
               ++it) {
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
          }
          // else {
          //   matched = false;
          //   VLOG(6) << " --- match false: " << drr_child_op->name()
          //           << " not found.";
          //   break;
          // }
        }
      }

      step++;
    }

    if (matched) {
      VLOG(6) << "step: " << step
              << " CountOfOpCalls: " << source_pattern_graph_->CountOfOpCalls();
      IR_ENFORCE(step == source_pattern_graph_->CountOfOpCalls(),
                 "step not equal to count of OpCalls");
    } else {
      VLOG(6) << " --- match false: " << op->name();
      return matched;
    }
    // matched = matched && step == source_pattern_graph_->CountOfOpCalls();

    // Constraints
    MatchContext match_context{source_pattern_match_ctx};
    for (const auto& constraint : constraints_) {
      matched = constraint(match_context);
      if (!matched) {
        VLOG(6) << " --- match false: constraint not satisfied";
        break;
      }
    }

    return matched;
  }

  std::unordered_map<const OpCall*, std::unordered_set<pir::Operation*>>
  FindCandidateIrOutputOp(
      pir::Operation* op,
      const OpCall* anchor,
      const SourcePatternGraph& source_pattern_graph) const {
    // get all source pattern op
    std::unordered_set<const OpCall*> drr_output_op_set;
    auto id2tensor = source_pattern_graph.id2owend_tensor();
    for (auto output_tensor_name : source_pattern_graph.output_tensors()) {
      OpCall* output_op_candidate =
          id2tensor[output_tensor_name].get()->producer();
      if (std::all_of(output_op_candidate->outputs().begin(),
                      output_op_candidate->outputs().end(),
                      [&source_pattern_graph](const Tensor* output) -> bool {
                        return source_pattern_graph.output_tensors().count(
                            output->name());
                      })) {
        drr_output_op_set.insert(output_op_candidate);
      }
    }
    std::unordered_map<const OpCall*, std::unordered_set<pir::Operation*>>
        output_op_bind_map{{anchor, {op}}};
    if (drr_output_op_set.size() == 1) {
      return output_op_bind_map;
    }
    std::unordered_set<const OpCall*> drr_visited_ops{anchor};
    DfsVisitor(
        anchor, op, drr_output_op_set, &drr_visited_ops, &output_op_bind_map);
    if (output_op_bind_map.size() != drr_output_op_set.size()) {
      return {};
    }
    return output_op_bind_map;
  }

  void DfsVisitor(
      const OpCall* drr_op,
      pir::Operation* ir_op,
      const std::unordered_set<const OpCall*>& drr_output_op_set,
      std::unordered_set<const OpCall*>* drr_visited_ops,
      std::unordered_map<const OpCall*, std::unordered_set<pir::Operation*>>*
          output_op_bind_map) const {
    VLOG(6) << "DfsVisitor Start: drr op(" << drr_op->name() << ")"
            << "ir op(" << ir_op->name() << ")";
    if (drr_op->name() != ir_op->name()) {
      return;
    }
    // check input's size
    const auto& drr_op_input_tensors = drr_op->inputs();
    auto ir_op_input_value_size = ir_op->num_operands();
    if (drr_op_input_tensors.size() != ir_op_input_value_size) {
      return;
    }
    // check output's size
    const auto& drr_op_output_tensors = drr_op->outputs();
    auto ir_op_output_value_size = ir_op->num_results();
    if (drr_op_output_tensors.size() != ir_op_output_value_size) {
      return;
    }
    // check producer op
    for (size_t i = 0; i < drr_op_input_tensors.size(); ++i) {
      // case 1: drr_op_input_tensor is the input tensor of source pattern
      if (drr_op_input_tensors[i]->producer() == nullptr) {
        // dfs source pattern input tensor other child op
        auto ir_input_tensor = ir_op->operand(i).source();
        for (auto drr_bro_op : drr_op_input_tensors[i]->consumers()) {
          if (drr_visited_ops->count(drr_bro_op)) {
            continue;
          }
          for (auto it = ir_input_tensor.use_begin();
               it != ir_input_tensor.use_end();
               ++it) {
            auto* ir_bro_op = it.owner();
            if (drr_bro_op->name() == ir_bro_op->name()) {
              drr_visited_ops->insert(drr_bro_op);
              DfsVisitor(drr_bro_op,
                         ir_bro_op,
                         drr_output_op_set,
                         drr_visited_ops,
                         output_op_bind_map);
              drr_visited_ops->erase(drr_bro_op);
            }
          }
        }
        continue;
      }
      // case 2: have producer op
      const auto& drr_producer_op = drr_op_input_tensors[i]->producer();
      if (drr_visited_ops->count(drr_producer_op)) {
        continue;
      }
      auto ir_operand_value = ir_op->operand(i).source();
      if (drr_op_input_tensors[i]->consumers().size() !=
          ir_operand_value.use_count()) {
        return;
      }
      auto* ir_producer_op = ir_operand_value.GetDefiningOp();
      drr_visited_ops->insert(drr_producer_op);
      DfsVisitor(drr_producer_op,
                 ir_producer_op,
                 drr_output_op_set,
                 drr_visited_ops,
                 output_op_bind_map);

      drr_visited_ops->erase(drr_producer_op);
    }
    if (drr_output_op_set.count(drr_op)) {
      (*output_op_bind_map)[drr_op].insert(ir_op);
      return;
    }
    // check child ops
    for (size_t i = 0; i < drr_op_output_tensors.size(); ++i) {
      const auto& drr_child_ops = drr_op_output_tensors[i]->consumers();
      auto ir_output_value = ir_op->result(i);
      if (drr_child_ops.size() != ir_output_value.use_count()) {
        return;
      }
      for (auto* drr_child_op : drr_child_ops) {
        for (auto it = ir_output_value.use_begin();
             it != ir_output_value.use_end();
             ++it) {
          auto* ir_child_op = it.owner();
          if (drr_child_op->name() == ir_child_op->name()) {
            if (drr_visited_ops->count(drr_child_op)) {
              continue;
            }
            drr_visited_ops->insert(drr_child_op);
            DfsVisitor(drr_child_op,
                       ir_child_op,
                       drr_output_op_set,
                       drr_visited_ops,
                       output_op_bind_map);
            drr_visited_ops->erase(drr_child_op);
          }
        }
      }
    }  // check child ops
    return;
  }

  bool MatchFromOutputToInput(
      std::vector<const OpCall*> drr_output_sequence,
      std::vector<pir::Operation*> ir_output_sequence,
      const SourcePatternGraph& source_pattern_graph,
      const std::shared_ptr<MatchContextImpl>& source_pattern_match_ctx) const {
    VLOG(6) << "MatchFromOutputToInput Start";
    IR_ENFORCE(drr_output_sequence.size() == ir_output_sequence.size());
    std::unordered_set<const OpCall*> drr_visited;
    std::unordered_set<Operation*> ir_visited;
    std::queue<const OpCall*> drr_q;
    std::queue<pir::Operation*> ir_q;
    bool matched = true;
    size_t step = 0;
    for (size_t i = 0; i < ir_output_sequence.size(); ++i) {
      drr_q.push(drr_output_sequence[i]);
      drr_visited.insert(drr_output_sequence[i]);
      ir_q.push(ir_output_sequence[i]);
      ir_visited.insert(ir_output_sequence[i]);
    }
    while (!drr_q.empty()) {
      if (!matched) break;
      auto* drr_node = drr_q.front();
      auto* ir_node = ir_q.front();
      drr_q.pop();
      ir_q.pop();
      if (drr_node->name() != ir_node->name()) {
        matched = false;
        break;
      }
      const auto& drr_input_tensors = drr_node->inputs();
      auto ir_input_value_size = ir_node->num_operands();
      if (drr_input_tensors.size() != ir_input_value_size) {
        matched = false;
        break;
      }
      if (drr_node->outputs().size() != ir_node->num_results()) {
        matched = false;
        break;
      }
      source_pattern_match_ctx->BindIrOperation(
          drr_node, std::make_shared<IrOperation>(ir_node));
      // binding input_tensor of current_op
      for (size_t i = 0; i < drr_input_tensors.size(); ++i) {
        source_pattern_match_ctx->BindIrValue(
            drr_input_tensors[i]->name(),
            std::make_shared<IrValue>(ir_node->operand(i).source()));
        auto* drr_producer_op = drr_input_tensors[i]->producer();
        auto* ir_producer_op = ir_node->operand(i).source().GetDefiningOp();
        if (drr_producer_op == nullptr) {
          continue;
        }
        if (drr_input_tensors[i]->consumers().size() !=
            ir_node->operand(i).source().use_count()) {
          matched = false;
          break;
        }
        // bfs producer_op of current_op
        if (!drr_visited.count(drr_producer_op)) {
          drr_q.push(drr_producer_op);
          ir_q.push(ir_producer_op);
          drr_visited.insert(drr_producer_op);
          ir_visited.insert(ir_producer_op);
        }
      }
      // binding output tensor of current_op
      auto drr_op_output_tensor = drr_node->outputs();
      for (size_t j = 0; j < drr_op_output_tensor.size(); j++) {
        source_pattern_match_ctx->BindIrValue(
            drr_op_output_tensor[j]->name(),
            std::make_shared<IrValue>(ir_node->result(j)));
      }
      ++step;
    }

    if (matched) {
      IR_ENFORCE(step == source_pattern_graph.CountOfOpCalls());
    } else {
      return matched;
    }

    MatchContext match_context{source_pattern_match_ctx};
    for (const auto& constraint : constraints_) {
      matched = constraint(match_context);
      if (!matched) break;
    }

    return matched;
  }

  bool PatternGraphMatchV2(pir::Operation* op,
                           MatchContextImpl* source_pattern_match_ctx) const {
    VLOG(6) << "PatternGraphMatch Start: op(" << op->name() << ")";
    const OpCall* anchor = source_pattern_graph_->AnchorNode();
    std::unordered_map<const OpCall*, std::unordered_set<pir::Operation*>>
        bind_map =
            FindCandidateIrOutputOp(op, anchor, *(source_pattern_graph_.get()));
    if (bind_map.empty()) {
      return false;
    }
    std::vector<const OpCall*> drr_output_sequence;
    std::vector<Operation*> ir_output_sequence;

    for (auto pair : bind_map) {
      drr_output_sequence.push_back(pair.first);
    }
    // using dfs to obtain the arrangement of all candidate ir ops
    auto permute = [&](auto&& permute, size_t index) -> bool {
      if (index == drr_output_sequence.size()) {
        // new match_ctx
        std::shared_ptr<MatchContextImpl> match_ctx =
            std::make_shared<MatchContextImpl>();
        if (MatchFromOutputToInput(drr_output_sequence,
                                   ir_output_sequence,
                                   *(source_pattern_graph_.get()),
                                   match_ctx)) {
          *source_pattern_match_ctx = *match_ctx;
          return true;
        }
        return false;
      }
      for (auto* ir_op : bind_map[drr_output_sequence[index]]) {
        ir_output_sequence.push_back(ir_op);
        if (permute(permute, index + 1)) {
          return true;
        }
        ir_output_sequence.pop_back();
      }
      return false;
    };

    return permute(permute, 0);
  }

  void PatternGraphRewrite(const MatchContextImpl& source_pattern_match_ctx,
                           pir::PatternRewriter& rewriter) const {  // NOLINT
    VLOG(6) << "Create Operations in result_pattern_graph";
    MatchContextImpl res_match_ctx = CreateOperations(*source_pattern_graph_,
                                                      *result_pattern_graph_,
                                                      source_pattern_match_ctx,
                                                      rewriter);
    VLOG(6) << "Process Assign Tensor";
    RebindIrTensorForAssignTensor(*result_pattern_graph_, &res_match_ctx);
    VLOG(6) << "Replace Output Values in source_pattern_graph by Output Values "
               "in result_pattern_graph";
    ReplaceOutputTensor(source_pattern_match_ctx, res_match_ctx, rewriter);
    VLOG(6) << "Delete Operations in source_pattern_graph";
    DeleteSourcePatternOp(*source_pattern_graph_,
                          *result_pattern_graph_,
                          source_pattern_match_ctx,
                          rewriter);
  }

 private:
  MatchContextImpl CreateOperations(
      const SourcePatternGraph& source_pattern_graph,
      const ResultPatternGraph& result_pattern_graph,
      const MatchContextImpl& src_match_ctx,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    MatchContextImpl res_match_ctx;
    // add input tensors info for res_match_ctx
    for (const auto& in_tensor : result_pattern_graph.input_tensors()) {
      IR_ENFORCE(result_pattern_graph.id2owend_tensor().count(in_tensor),
                 "Drr input tensor [%s] must exists in result pattern graph.",
                 in_tensor);
      if (!result_pattern_graph.id2owend_tensor().at(in_tensor)->is_none()) {
        res_match_ctx.BindIrValue(
            in_tensor,
            std::make_shared<IrValue>(src_match_ctx.GetIrValue(in_tensor)));
      }
    }
    // set insert point
    for (const auto& output : result_pattern_graph.output_tensors()) {
      if (source_pattern_graph.id2owend_tensor().count(output)) {
        auto ir_value = src_match_ctx.GetIrValue(output);
        if (ir_value.get()) {
          rewriter.SetInsertionPointAfter(ir_value.get().GetDefiningOp());
          break;
        }
      }
    }
    // topo order visit result_pattern_graph
    GraphTopo graph_topo_visit(&result_pattern_graph);
    graph_topo_visit.WalkGraphNodesTopoOrder([&](const OpCall& op_call) {
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
                           pir::PatternRewriter& rewriter) const {  // NOLINT
    for (const auto& output_name : result_pattern_graph_->output_tensors()) {
      if (source_pattern_graph_->output_tensors().count(output_name)) {
        const auto& src_ir_tensor = src_match_ctx.GetIrValue(output_name);
        const auto& res_ir_tensor = res_match_ctx.GetIrValue(output_name);
        rewriter.ReplaceAllUsesWith(src_ir_tensor.get(), res_ir_tensor.get());
      } else {
        LOG(WARNING) << "The output tensor (" << output_name
                     << ") in the result_pattern_graph is not the tensor"
                        " in source_pattern_graph.";
      }
    }
  }

  void DeleteSourcePatternOp(const SourcePatternGraph& source_pattern_graph,
                             const ResultPatternGraph& result_pattern_graph,
                             const MatchContextImpl& src_match_ctx,
                             pir::PatternRewriter& rewriter) const {  // NOLINT
    std::vector<const OpCall*> topo_order_ops;
    GraphTopo graph_topo_visit(&source_pattern_graph);
    graph_topo_visit.WalkGraphNodesTopoOrder(
        [&topo_order_ops](const OpCall& op_call) {
          topo_order_ops.push_back(&op_call);
        });

    // Filter the operations which are replaced by result pattern
    // 1. Filter operations by forward walk
    std::unordered_set<std::string> forward_visited_tensor_set(
        result_pattern_graph.input_tensors());
    std::unordered_set<const OpCall*> forward_deleted_ops;
    std::for_each(topo_order_ops.begin(),
                  topo_order_ops.end(),
                  [&forward_deleted_ops,
                   &forward_visited_tensor_set](const OpCall* op_call) {
                    if (op_call->inputs().empty()) {
                      forward_deleted_ops.insert(op_call);
                      for (const auto* output : op_call->outputs()) {
                        forward_visited_tensor_set.insert(output->name());
                      }
                    }
                    for (const auto* input : op_call->inputs()) {
                      if (forward_visited_tensor_set.count(input->name())) {
                        forward_deleted_ops.insert(op_call);
                        for (const auto* output : op_call->outputs()) {
                          forward_visited_tensor_set.insert(output->name());
                        }
                        break;
                      }
                    }
                  });
    // 2. Filter operations by backward walk and merge the forward result
    std::unordered_set<std::string> backward_visited_tensor_set(
        result_pattern_graph.output_tensors());
    std::vector<const OpCall*> deleted_ops;
    std::unordered_set<const OpCall*> deleted_ops_set;
    std::for_each(
        topo_order_ops.rbegin(),
        topo_order_ops.rend(),
        [&deleted_ops,
         &deleted_ops_set,
         &backward_visited_tensor_set,
         &forward_deleted_ops](const OpCall* op_call) {
          bool all_comsumer_deleted = true;
          for (const auto* output : op_call->outputs()) {
            if (backward_visited_tensor_set.count(output->name())) {
              for (const auto* consumer : output->consumers()) {
                if (!deleted_ops_set.count(consumer)) {
                  all_comsumer_deleted = false;
                }
              }
            } else if (output->consumers().empty()) {
              continue;
            } else {
              all_comsumer_deleted = false;
            }
          }
          if (all_comsumer_deleted && forward_deleted_ops.count(op_call)) {
            deleted_ops_set.insert(op_call);
            deleted_ops.push_back(op_call);
            for (const auto* input : op_call->inputs()) {
              backward_visited_tensor_set.insert(input->name());
            }
          }
        });

    // Delete Operation with topo order from output tensors.
    for (const auto* op_call : deleted_ops) {
      IR_ENFORCE(src_match_ctx.operation_map().count(op_call),
                 "Drr OpCall [%s] must exists in match context.",
                 op_call->name());
      auto* op = src_match_ctx.operation_map().at(op_call)->get();
      VLOG(6) << "Delete (" << op_call->name() << " @" << op_call << " :@" << op
              << ") in source_pattern_graph ";
      rewriter.EraseOp(op);
    }
  }
  const std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  const std::vector<Constraint> constraints_;
  const std::shared_ptr<ResultPatternGraph> result_pattern_graph_;
};

}  // namespace drr
}  // namespace pir
