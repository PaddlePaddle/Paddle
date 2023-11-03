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

#include "paddle/fluid/pir/drr/drr_rewrite_pattern.h"

namespace pir {
namespace drr {

bool DrrRewritePattern::MatchAndRewrite(
    pir::Operation* op,
    PatternRewriter& rewriter) const {  // NOLINT
  std::shared_ptr<MatchContextImpl> src_match_ctx =
      std::make_shared<MatchContextImpl>();
  if (PatternGraphMatch(op, src_match_ctx.get())) {
    VLOG(4) << "DRR pattern (" << pattern_name_ << ") is matched in program.";
    PatternGraphRewrite(*src_match_ctx, rewriter);
    return true;
  }
  return false;
}

bool DrrRewritePattern::PatternGraphMatch(
    pir::Operation* op, MatchContextImpl* source_pattern_match_ctx) const {
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
  std::unordered_map<const OpCall*, Operation*> output_op_map;
  for (auto pair : bind_map) {
    drr_output_sequence.push_back(pair.first);
  }
  // using dfs to obtain the arrangement of all candidate ir ops
  auto permute = [&](auto&& permute, size_t index) -> bool {
    if (index == drr_output_sequence.size()) {
      // avoiding duplicate binding of ir op
      std::unordered_set<Operation*> ir_output_set;
      for (Operation* op : ir_output_sequence) {
        auto pr = ir_output_set.insert(op);
        if (pr.second == false) {
          return false;
        }
      }
      // new match_ctx
      std::shared_ptr<MatchContextImpl> match_ctx =
          std::make_shared<MatchContextImpl>();
      std::transform(drr_output_sequence.begin(),
                     drr_output_sequence.end(),
                     ir_output_sequence.begin(),
                     std::inserter(output_op_map, output_op_map.end()),
                     [](const OpCall* drr_op, Operation* ir_op) {
                       return std::make_pair(drr_op, ir_op);
                     });
      if (MatchFromOutputToInput(
              output_op_map, *(source_pattern_graph_.get()), match_ctx)) {
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

std::unordered_map<const OpCall*, std::unordered_set<pir::Operation*>>
DrrRewritePattern::FindCandidateIrOutputOp(
    pir::Operation* op,
    const OpCall* anchor,
    const SourcePatternGraph& source_pattern_graph) const {
  // get source pattern output op
  std::unordered_set<const OpCall*> drr_output_op_set =
      source_pattern_graph.OutputNodes();
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

void DrrRewritePattern::DfsVisitor(
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
  // check op input's size
  const auto& drr_op_input_tensors = drr_op->inputs();
  auto ir_op_input_value_size = ir_op->num_operands();
  if (drr_op_input_tensors.size() != ir_op_input_value_size) {
    return;
  }
  // check op output's size
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
    auto* ir_producer_op = ir_operand_value.dyn_cast<pir::OpResult>().owner();
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

bool DrrRewritePattern::MatchFromOutputToInput(
    std::unordered_map<const OpCall*, Operation*> output_op_map,
    const SourcePatternGraph& source_pattern_graph,
    const std::shared_ptr<MatchContextImpl>& source_pattern_match_ctx) const {
  VLOG(6) << "MatchFromOutputToInput Start";
  std::unordered_set<const OpCall*> drr_visited;
  std::unordered_set<Operation*> ir_visited;
  std::queue<const OpCall*> drr_q;
  std::queue<pir::Operation*> ir_q;
  bool matched = true;
  size_t step = 0;
  for (auto it = output_op_map.begin(); it != output_op_map.end(); ++it) {
    VLOG(6) << "match (" << it->first->name() << " @" << it->first << " : @"
            << it->second << ") in source_pattern_graph ";
    drr_q.push(it->first);
    drr_visited.insert(it->first);
    ir_q.push(it->second);
    ir_visited.insert(it->second);
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
      if (drr_producer_op == nullptr) {
        continue;
      }
      auto* ir_producer_op =
          ir_node->operand(i).source().dyn_cast<pir::OpResult>().owner();
      if (drr_input_tensors[i]->consumers().size() !=
          ir_node->operand(i).source().use_count()) {
        matched = false;
        break;
      }
      // bfs producer_op of current_op
      if (!drr_visited.count(drr_producer_op) &&
          !ir_visited.count(ir_producer_op)) {
        drr_q.push(drr_producer_op);
        ir_q.push(ir_producer_op);
        drr_visited.insert(drr_producer_op);
        ir_visited.insert(ir_producer_op);
      } else {
        matched = false;
        break;
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
    PADDLE_ENFORCE_EQ(
        step,
        source_pattern_graph.CountOfOpCalls(),
        phi::errors::PreconditionNotMet(
            "Pattern matching failed."
            "The number of successful matches and the number of OpCalls in the "
            "source pattern graph are not equal."));
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

void DrrRewritePattern::PatternGraphRewrite(
    const MatchContextImpl& source_pattern_match_ctx,
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

MatchContextImpl DrrRewritePattern::CreateOperations(
    const SourcePatternGraph& source_pattern_graph,
    const ResultPatternGraph& result_pattern_graph,
    const MatchContextImpl& src_match_ctx,
    pir::PatternRewriter& rewriter) const {  // NOLINT
  MatchContextImpl res_match_ctx;
  // add input tensors info for res_match_ctx
  for (const auto& in_tensor : result_pattern_graph.input_tensors()) {
    PADDLE_ENFORCE_NE(
        result_pattern_graph.id2owend_tensor().count(in_tensor),
        0,
        phi::errors::NotFound("Not found the input tensor."
                              "Drr input tensor [%s] must exist in the result "
                              "pattern graph to be obtained.",
                              in_tensor));
    if (!result_pattern_graph.id2owend_tensor().at(in_tensor)->is_none()) {
      res_match_ctx.BindIrValue(
          in_tensor,
          std::make_shared<IrValue>(src_match_ctx.GetIrValue(in_tensor)));
    }
  }

  if (result_pattern_graph.CountOfOpCalls() == 1) {
    CreateOperation(*result_pattern_graph.owned_op_call()[0],
                    src_match_ctx,
                    rewriter,
                    &res_match_ctx);
    return res_match_ctx;
  }

  std::vector<std::vector<Operation*>> temp_program;
  std::unordered_map<Operation*, size_t> op_2_temp_program_index;
  for (Operation* op : *rewriter.block()) {
    op_2_temp_program_index[op] = temp_program.size();
    temp_program.push_back({op});
  }

  // topo order visit result_pattern_graph
  GraphTopo graph_topo_visit(&result_pattern_graph);
  graph_topo_visit.WalkGraphNodesTopoOrder([&](const OpCall& op_call) {
    // set insert point
    size_t max_input_op_index = 0;
    Operation* max_index_op = nullptr;
    for (const Tensor* input : op_call.inputs()) {
      if (input->is_none()) {
        continue;
      }
      Value ir_val = res_match_ctx.GetIrValue(input->name()).get();
      if (ir_val) {
        Operation* ir_input_op = ir_val.dyn_cast<pir::OpResult>().owner();
        if (max_input_op_index < op_2_temp_program_index[ir_input_op]) {
          max_input_op_index = op_2_temp_program_index[ir_input_op];
          max_index_op = ir_input_op;
        } else if (max_input_op_index == op_2_temp_program_index[ir_input_op]) {
          const auto& ops_vec = temp_program[max_input_op_index];
          for (auto it = ops_vec.rbegin(); it != ops_vec.rend(); it++) {
            if (*it == max_index_op) {
              break;
            } else if (*it == ir_input_op) {
              max_index_op = ir_input_op;
              break;
            } else {
              // do nothing
            }
          }
        } else {
          // do nothing
        }
      }
    }
    if (max_input_op_index == 0UL) {
      VLOG(6) << "Not found producer op for (" << op_call.name() << ")";
      Operation* source_patter_first_op =
          src_match_ctx.Operation(source_pattern_graph.owned_op_call()[0].get())
              .get();
      max_input_op_index = op_2_temp_program_index[source_patter_first_op];
      rewriter.SetInsertionPoint(source_patter_first_op);
    } else {
      rewriter.SetInsertionPointAfter(max_index_op);
    }

    Operation* new_op =
        CreateOperation(op_call, src_match_ctx, rewriter, &res_match_ctx);
    op_2_temp_program_index[new_op] = max_input_op_index + 1;
    temp_program[max_input_op_index + 1].push_back(new_op);
  });

  return res_match_ctx;
}

void DrrRewritePattern::RebindIrTensorForAssignTensor(
    const ResultPatternGraph& result_pattern_graph,
    MatchContextImpl* res_match_ctx) const {
  const auto& tensor_assign_map = result_pattern_graph.tensor_assign_map();
  for (const auto& kv : tensor_assign_map) {
    const auto& src_tensor_name = kv.first;
    const auto& dst_tensor_name = kv.second;
    res_match_ctx->BindIrValue(
        src_tensor_name,
        std::make_shared<IrValue>(res_match_ctx->GetIrValue(dst_tensor_name)));
  }
}

void DrrRewritePattern::ReplaceOutputTensor(
    const MatchContextImpl& src_match_ctx,
    const MatchContextImpl& res_match_ctx,
    pir::PatternRewriter& rewriter) const {  // NOLINT
  for (const auto& output_name : result_pattern_graph_->output_tensors()) {
    if (source_pattern_graph_->id2owend_tensor().count(output_name)) {
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

void DrrRewritePattern::DeleteSourcePatternOp(
    const SourcePatternGraph& source_pattern_graph,
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
  std::for_each(topo_order_ops.rbegin(),
                topo_order_ops.rend(),
                [&deleted_ops,
                 &deleted_ops_set,
                 &backward_visited_tensor_set,
                 &forward_deleted_ops](const OpCall* op_call) {
                  bool all_comsumer_deleted = true;
                  bool from_backward_visited_tensor = false;
                  for (const auto* output : op_call->outputs()) {
                    if (backward_visited_tensor_set.count(output->name())) {
                      from_backward_visited_tensor = true;
                    } else if (output->consumers().empty()) {
                      continue;
                    } else {
                      all_comsumer_deleted = false;
                    }
                  }
                  if (all_comsumer_deleted && from_backward_visited_tensor &&
                      forward_deleted_ops.count(op_call)) {
                    deleted_ops_set.insert(op_call);
                    deleted_ops.push_back(op_call);
                    for (const auto* input : op_call->inputs()) {
                      backward_visited_tensor_set.insert(input->name());
                    }
                  }
                });

  // Delete Operation with topo order from output tensors.
  for (const auto* op_call : deleted_ops) {
    PADDLE_ENFORCE_NE(src_match_ctx.operation_map().count(op_call),
                      0,
                      phi::errors::NotFound(
                          "Not found the OpCall."
                          "Only Opcall [%s] that exist in match context can be "
                          "deleted.",
                          op_call->name()));
    auto* op = src_match_ctx.operation_map().at(op_call)->get();
    VLOG(6) << "Delete (" << op_call->name() << " @" << op_call << " :@" << op
            << ") in source_pattern_graph ";
    rewriter.EraseOp(op);
  }
}

}  // namespace drr
}  // namespace pir
