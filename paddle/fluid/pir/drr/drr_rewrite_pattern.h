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
    if (PatternGraphMatchV2(op, src_match_ctx)) {
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
    std::unordered_set<Operation*> ir_visited;
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
                 "step not equal to count of opcalls");
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

  std::unordered_map<const OpCall*, std::unordered_set<const pir::Operation*>>
  FindOutputOp(pir::Operation* op) const {
    // get anchor
    OpCall* anchor = source_pattern_graph_->AnchorNode();
    // get drr_output op
    std::unordered_set<const Opcall*> drr_output_op;
    auto id2tensor = source_pattern_graph_->id2owend_tensor();
    for (auto output_tensor_name : source_pattern_graph_->output_tensors()) {
      drr_output_op.insert(id2tensor[output_tensor_name].get()->producer());
    }
    std::unordered_map<const OpCall*, std::unordered_set<const pir::Operation*>>
        output_op_bind_map;
    std::unordered_set<OpCall*> drr_visited;
    auto dfs = [&](auto&& dfs, OpCall* drr_op, pir::Operation* ir_op) -> bool {
      if (drr_visited.count(drr_op)) {
        VLOG(6) << "drr_op has already been visited.";
        return false;
      }
      drr_visited.insert(drr_op);
      if (drr_op->name() != ir_op->name()) {
        VLOG(6) << "drr_op and ir_op have different op names. "
                << "drr_op name :" << drr_op->name()
                << "ir_op name :" << ir_op->name();
        return false;
      }
      // check input's size
      const auto& drr_op_input_tensors = drr_op->inputs();
      auto ir_op_input_value_size = ir_op->num_operands();
      if (drr_input_tensors.size() != ir_input_value_size) {
        VLOG(6) << "drr_op and ir_op have different input size. "
                << "drrOp_input_tensors.size():" << drr_input_tensors.size()
                << "ir_input_value_size:" << ir_input_value_size;
        return false;
      }
      // check output's size
      const auto& drr_op_output_tensors = drr_op->outputs();
      auto ir_op_output_value_size = ir_op->num_results()
          : if (drr_op_output_tensors.size() != ir_op_output_value_size) {
        VLOG(6) << "drr_op and ir_op have different input size. "
                << "drrOp_input_tensors.size():" << drr_input_tensors.size()
                << "ir_input_value_size:" << ir_input_value_size;
        return false;
      }
      // bianing drr_output and ir_output_candidate
      if (drr_output_op.count(drr_op)) {
        output_op_bind_map[drr_op].insert(ir_op);
        return true;
      }
      // check child ops 优先向下找
      for (size_t i = 0; i < drr_op_output_tensors.size(); ++i) {
        const auto& drr_child_ops = drr_op_output_tensors[i]->consumers();
        auto ir_output_value = ir_op->result(i).source();
        if (drr_child_ops.size() != ir_output_value.use_count()) {
          VLOG(6) << " drr_output_tensor and ir_output_tensor have different "
                     "consumer number."
                  << "drr_op_output_tensors[" << i
                  << "]: " << drr_child_ops.size() << "ir_op->result(" << i
                  << "): " << ir_output_value.use_count();
          return false;
        }
        for (auto* drr_child_op : drr_child_ops) {
          bool flag = false;
          for (auto it = ir_output_value.ues_begin();
               it != ir_output_value.use_end();
               ++it) {
            auto* ir_child_op = it.owner();
            // 重名op 全遍历
            if (drr_child_op->name() == ir_child_op->name()) {
              flag = true;
              dfs(dfs, drr_child_op, ir_child_op);
            }
          }
          if (!flag) {
            VLOG(6) << "no type name matching the child node of drr_op among "
                       "the child nodes of ir_op. "
                    << "drr_child_op name : " << drr_child_op->name();
            return false;
          }
        }
      }

      // check producer op
      for (size_t i = 0; i < drr_op_input_tensors.size(); ++i) {
        // drr_op_input_tensor is the input tensor of source pattern
        if (source_pattern_graph_->input_tensors().count(
                drr_op_input_tensors[i]->name())) {
          continue;
        }
        const auto& drr_producer_op = drr_op_input_tensors[i]->producer();
        auto ir_operand_value = ir_op->operand(i).source();
        if (drr_op_input_tensors[i]->consumers().size() !=
            ir_operand_value.use_count()) {
          VLOG(6) << "input tensor have diff num of consumer "
                  << "drr_op_input_tensors consumers : "
                  << drr_op_input_tensors[i]->consumers().size()
                  << "ir_operand_value use_count : "
                  << ir_operand_value.use_count();
          return false;
        }

        if (drr_visited.count(drr_producer_op)) {
          VLOG(6) << "drr_producer_op has already been visited.";
          return false;
        }

        auto* ir_producer_op = ir_operand_value.GetDefiningOp();
        dfs(dfs, drr_producer_op, ir_producer_op);
      }
    };

    dfs(dfs, anchor, op);

    // TODO(gst): source_pattern 的outputop是否都找到了对应的ir_op
    if (output_op_bind_map.size() != drr_output_op.size()) {
      VLOG(6) << "未匹配到source pattern中所有的output op";
      return nullptr;
    }
    return output_op_bind_map;
  }

  bool MatchFromBackToFront(
      std::vector<OpCall*> drr_output_sequence,
      std::vector<pir::Operation*> ir_output_sequence,
      const std::shared_ptr<MatchContextImpl>& source_pattern_match_ctx) const {
    VLOG(6) << "Assert drr_output and ir_output have equal lengths" IR_ENFORCE(
        drr_output_sequence.size() == ir_output_sequence.size());
    // init
    std::unordered_set<const OpCall*> drr_visited;
    std::unordered_set<Operation*> ir_visited;
    std::queue<const OpCall*> drr_q;
    std::queue<pir::Operation*> ir_q;
    bool matched = true;
    for (size_t i = 0; i < ir_output_sequence.size(); ++i) {
      drr_q.push(drr_output_sequence[i]);
      drr_visited.insert(drr_output_sequence[i]);
      ir_q.push(ir_output_sequence[i]);
      ir_visited.insert(ir_output_sequence[i]);
      source_pattern_match_ctx->BindIrOperation(
          drr_output_sequence[i],
          std::make_shared<IrOperation>(ir_output_sequence[i]));
    }
    size_t step = 0;
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
      // check input size
      if (drr_input_tensors.size() != ir_input_value_size) {
        VLOG(6) << "Match False! drr_node input size:"
                << drr_input_tensors.size()
                << "not equal ir_node input size:" << ir_input_value_size;
        matched = false;
        break;
      }
      // check output size
      if (drr_node->outputs().size() != ir_node->num_results()) {
        VLOG(6) << "Match False! drr_node output size:"
                << drr_node->outputs().size()
                << "not equal ir_node output size:" << ir_node->num_results();
        matched = false;
        break;
      }
      // check visited
      if (drr_visited.count(drr_node) && ir_visited.count(ir_node)) {
        continue;
      } else if (!(!drr_visited.count(drr_node) &&
                   !ir_visited.count(ir_node))) {
        VLOG(6) << "binding of ir_node and drr_node is not synchronized";
        matched = false;
        break;
      }
      source_pattern_match_ctx->BindIrOperation(
          drr_node, std::make_shared<IrOperation>(ir_node));
      // join the producerOp of input
      for (size_t i = 0; i < drr_input_tensors.size(); ++i) {
        auto* drr_producer_op = drr_input_tensors[i]->producer();
        auto* ir_producer_op = ir_node->operand(i).source().GetDefiningOp();
        if (drr_producer_op->name() != ir_producer_op->name()) {
          VLOG(6) << "Match False! drr_producer_op name:"
                  << drr_node->outputs().size()
                  << "not equal ir_producer_op node:" << ir_node->num_results();
          matched = false;
          break;
        } else {
          drr_q.push(drr_producer_op);
          ir_q.push(ir_producer_op);
          drr_visited.insert(drr_producer_op);
          ir_visited.insert(ir_producer_op);
        }
      }

      ++step;
    }

    if (matched) {
      IR_ENFORCE(step == source_pattern_graph_->CountOfOpCalls());
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

  bool PatternGraphMatchV2(
      pir::Operation* op,
      const std::shared_ptr<MatchContextImpl>& source_pattern_match_ctx) const {
    std::unordered_map<const OpCall*, std::unordered_set<const pir::Operation*>>
        bind_map = FindOutputOp(op, source_pattern_match_ctx);
    vector<Opcall*> drr_output_sequence;
    vector<Operation*> ir_output_sequence;
    vector<vector<pir::Operation*>> ir_output_sequence_candidate;
    for (auto it = bind_map.begin(); it != bind_map.end();) {
      drr_output_sequence.push_back(it->first);
      if (it->second.size() == 1) {
        ir_output_sequence.push_back(it->second);
        it = bind_map.erase(it);
      } else {
        ++it;
      }
    }
    auto permute =
        [&](auto&& permute,
            std::unordered_map<
                const OpCall*,
                std::unordered_set<const pir::Operation*>>::iterator iter) {
          if (iter == bind_map.end()) {
            ir_output_sequence_candidate.push_back(ir_output_sequence);
            return;
          }
          for (auto it = iter->second.begin(); it != iter->second.end(); ++it) {
            ir_output_sequence.push_back(it->first);
            permute(permute, ++iter);
            ir_output_sequence.pop_back();
          }
        };
    permute(permute, bind_map.begin());
    for (auto ir_candidate : ir_output_sequence_candidate) {
      // new match_ctx
      match_ctx = std::make_shared<MatchContextImpl>();
      if (MatchFromBackToFront(drr_output_sequence, ir_candidate, match_ctx)) {
        source_pattern_match_ctx = match_ctx;
        return true;
      }
    }
    return false;
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
