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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/cinn_group_lowering_pass.h"

#include <unordered_map>

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/common/broadcast_tree.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_pass.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"

#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"

PD_DECLARE_bool(cinn_enable_map_expr);

namespace {

using ShapeOrDataDimExprs4ValueT =
    std::function<const symbol::ShapeOrDataDimExprs&(pir::Value)>;

bool SameInputOutputShape(
    paddle::dialect::ExpandOp expand_op,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  const auto& x = ShapeOrDataDimExprs4Value(expand_op.x());
  const auto& shape = ShapeOrDataDimExprs4Value(expand_op.shape());
  const auto& out = ShapeOrDataDimExprs4Value(expand_op.out());
  if (x.data().has_value()) return false;
  if (!shape.data().has_value()) return false;
  if (out.data().has_value()) return false;
  CHECK(shape.data().value() == out.shape());
  return x.shape() == out.shape();
}

void ReplaceAllUsesWithInput(paddle::dialect::ExpandOp expand) {
  pir::Value x = expand.x();
  expand.out().ReplaceAllUsesWith(x);
}

// Returns true if success
bool EraseOneExpand(
    pir::Block* block,
    pir::PatternRewriter& rewriter,  // NOLINT
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  for (auto expand_it = block->begin(); expand_it != block->end();
       ++expand_it) {
    if (!expand_it->isa<paddle::dialect::ExpandOp>()) continue;
    auto expand = expand_it->dyn_cast<paddle::dialect::ExpandOp>();
    if (!SameInputOutputShape(expand, ShapeOrDataDimExprs4Value)) continue;
    auto generate_shape_op =
        expand.shape().defining_op<cinn::dialect::GenerateShapeOp>();
    CHECK_NOTNULL(generate_shape_op);
    ReplaceAllUsesWithInput(expand);
    rewriter.EraseOp(expand);
    rewriter.EraseOp(generate_shape_op);
    return true;
  }
  return false;
}

void EraseExpandsInBlock(
    pir::Block* block,
    pir::PatternRewriter& rewriter,  // NOLINT
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  while (EraseOneExpand(block, rewriter, ShapeOrDataDimExprs4Value)) {
    // Do nothing.
  }
}

void ReplaceExpandWithBroadcast(
    pir::IrContext* ir_context,
    pir::Block* block,
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  std::vector<pir::Operation*> op_list;
  for (auto& op : *block) {
    op_list.push_back(&op);
  }
  pir::Builder builder(ir_context, block);
  for (auto* op : op_list) {
    if (op && op->isa<paddle::dialect::ExpandOp>() &&
        op->operand_source(1)
            .defining_op()
            ->isa<cinn::dialect::GenerateShapeOp>()) {
      builder.SetInsertionPointAfter(op);
      auto broadcast =
          builder.Build<cinn::dialect::BroadcastOp>(op->operand_source(0),
                                                    std::vector<int64_t>{-1},
                                                    std::vector<int64_t>{-1});
      auto broadcast_out = broadcast.result(0);
      auto expand_out = op->result(0);
      expand_out.ReplaceAllUsesWith(broadcast_out);
      shape_analysis->SetShapeOrDataForValue(
          &broadcast_out, shape_analysis->GetShapeOrDataForValue(&expand_out));
      CHECK(op->use_empty());
      auto generate_shape_op = op->operand_source(1).defining_op();
      op->Erase();
      generate_shape_op->Erase();
    }
  }
}

std::vector<pir::Value> GetBlockOutsideInput(
    const std::vector<pir::Operation*> op_list) {
  std::vector<pir::Value> vec_res;
  std::unordered_set<::pir::Value> block_inner_output;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_results(); ++i) {
      block_inner_output.insert(op_list[k]->result(i));
    }
  }

  std::unordered_set<::pir::Value> insert_value;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_operands(); ++i) {
      if (!block_inner_output.count(op_list[k]->operand_source(i)) &&
          !insert_value.count(op_list[k]->operand_source(i))) {
        vec_res.push_back(op_list[k]->operand_source(i));
        insert_value.insert(op_list[k]->operand_source(i));
      }
    }
  }
  return vec_res;
}

std::vector<pir::Value> GetBlockOutsideOutput(
    const std::vector<pir::Operation*> op_list,
    const std::vector<pir::Operation*> group_all_list) {
  assert(group_all_list.size() >= 2);
  assert(group_all_list.back()->isa<pir::YieldOp>());

  auto yeild_op = group_all_list.back()->dyn_cast<pir::YieldOp>();

  std::unordered_set<pir::Value> yeild_inputs;
  for (size_t i = 0; i < yeild_op.num_operands(); ++i) {
    yeild_inputs.insert(yeild_op.operand_source(i));
  }

  std::unordered_set<pir::Operation*> innner_op_set(op_list.begin(),
                                                    op_list.end());
  std::unordered_set<pir::Operation*> outside_group_set;

  for (size_t i = 0; i < group_all_list.size(); ++i) {
    if (!innner_op_set.count(group_all_list[i])) {
      outside_group_set.insert(group_all_list[i]);
    }
  }

  std::vector<pir::Value> vec_res;

  for (auto* op : op_list) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      if (yeild_inputs.count(op->result(i))) {
        vec_res.push_back(op->result(i));
      } else {
        for (auto it = op->result(i).use_begin(); it != op->result(i).use_end();
             ++it) {
          if (outside_group_set.count(it->owner())) {
            vec_res.push_back(op->result(i));
            break;
          }
        }
      }
    }
  }
  return vec_res;
}

std::vector<pir::Operation*> GetOpListNotIncludeYield(
    const std::vector<pir::Operation*>& op_list) {
  std::vector<pir::Operation*> vec_res;
  for (size_t i = 0; i < op_list.size(); ++i) {
    if (!op_list[i]->isa<pir::YieldOp>()) {
      vec_res.push_back(op_list[i]);
    }
  }

  return vec_res;
}

std::vector<pir::Operation*> GetOutputOpList(
    const std::vector<pir::Operation*>& op_list) {
  std::vector<pir::Operation*> vec_res;
  auto yield_op = op_list.back();

  for (size_t i = 0; i < yield_op->num_operands(); ++i) {
    vec_res.push_back(
        yield_op->operand(i).source().dyn_cast<pir::OpResult>().owner());
  }

  return vec_res;
}

std::tuple<pir::Value, pir::Value, pir::Value> BroadcastableToCondValue(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis,
    const std::vector<pir::Value>& group_inputs,
    pir::Builder& builder) {  // NOLINT
  const auto& lhs_expr = broadcastable_condition->lhs;
  const auto& rhs_expr = broadcastable_condition->rhs;
  auto ShapeOrDataDimExprs4Value = [&shape_analysis](pir::Value value) {
    return shape_analysis->GetShapeOrDataForValue(&value);
  };

  std::vector<pir::Value> lhs_minial_inputs;
  std::vector<pir::Attribute> lhs_output_dim_expr_attrs;
  cinn::dialect::GenerateShapeOp::SymbolBindings lhs_symbol_bindings;
  bool success =
      cinn::dialect::MakeGenerateShapeOpAttribute(builder.ir_context(),
                                                  ShapeOrDataDimExprs4Value,
                                                  {lhs_expr},
                                                  group_inputs,
                                                  &lhs_minial_inputs,
                                                  &lhs_output_dim_expr_attrs,
                                                  &lhs_symbol_bindings);
  CHECK(success);
  std::vector<pir::Value> rhs_minial_inputs;
  std::vector<pir::Attribute> rhs_output_dim_expr_attrs;
  cinn::dialect::GenerateShapeOp::SymbolBindings rhs_symbol_bindings;
  success =
      cinn::dialect::MakeGenerateShapeOpAttribute(builder.ir_context(),
                                                  ShapeOrDataDimExprs4Value,
                                                  {rhs_expr},
                                                  group_inputs,
                                                  &rhs_minial_inputs,
                                                  &rhs_output_dim_expr_attrs,
                                                  &rhs_symbol_bindings);
  CHECK(success);

  auto lhs_value =
      builder
          .Build<cinn::dialect::GenerateShapeOp>(
              lhs_minial_inputs, lhs_output_dim_expr_attrs, lhs_symbol_bindings)
          .out();
  auto rhs_value =
      builder
          .Build<cinn::dialect::GenerateShapeOp>(
              rhs_minial_inputs, rhs_output_dim_expr_attrs, rhs_symbol_bindings)
          .out();

  auto const_one = builder
                       .Build<paddle::dialect::FullOp>(
                           std::vector<int64_t>{1}, 1, phi::DataType::INT64)
                       .out();
  auto lhs_eq_rhs_cond =
      builder.Build<paddle::dialect::EqualOp>(lhs_value, rhs_value).out();
  auto lhs_eq_one_cond =
      builder.Build<paddle::dialect::EqualOp>(lhs_value, const_one).out();
  auto rhs_eq_one_cond =
      builder.Build<paddle::dialect::EqualOp>(rhs_value, const_one).out();
  return std::tuple<pir::Value, pir::Value, pir::Value>(
      lhs_eq_rhs_cond, lhs_eq_one_cond, rhs_eq_one_cond);
}

cinn::dialect::ir::GroupPtr CloneGroup(const cinn::dialect::ir::GroupPtr& group,
                                       pir::Block* block,
                                       pir::IrMapping* ir_mapping) {
  auto new_group = group->Clone(block, *ir_mapping);
  new_group->shape_analysis = group->shape_analysis;
  return new_group;
}

void UpdateShapeAnalysis(
    const cinn::dialect::ir::GroupPtr& group,
    const pir::IrMapping& ir_mapping,
    const cinn::common::BroadcastLeaf& value_dim_exprs_list,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    pir::ShapeConstraintIRAnalysis* shape_analysis) {
  for (const auto& [origin_val, new_val] : ir_mapping.value_map()) {
    VLOG(1) << "#### UpdateShapeAnalysis origin_val: "
            << pir::GetValueId(&origin_val);
    VLOG(1) << "#### UpdateShapeAnalysis new_val: "
            << pir::GetValueId(&new_val);
    const auto& shape_dim_expr =
        value_dim_exprs_list->at(value_to_dim_expr_idx.at(origin_val));
    const auto& origin_shape_or_data =
        shape_analysis->GetShapeOrDataForValue(&origin_val);
    if (origin_shape_or_data.data()) {
      shape_analysis->SetShapeOrDataForValue(
          &new_val,
          symbol::ShapeOrDataDimExprs::MakeConsistentShapeOrData(
              shape_dim_expr));
    } else {
      shape_analysis->SetShapeOrDataForValue(
          &new_val, symbol::ShapeOrDataDimExprs{shape_dim_expr});
    }
  }
}

void SetLeafBlockByGroupView(
    const cinn::dialect::ir::GroupPtr& origin_group,
    const cinn::common::BroadcastLeaf& value_dim_exprs_list,
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    pir::Builder& builder,  // NOLINT
    pir::Block* block,
    std::unordered_map<pir::Block*, cinn::dialect::ir::GroupPtr>* group_map) {
  if (group_map->count(block)) {
    return;
  }

  pir::IrMapping ir_mapping;
  auto origin_group_inputs = GetBlockOutsideInput(origin_group->ops);
  for (auto input : origin_group_inputs) {
    ir_mapping.Add(input, input);
  }

  VLOG(1) << "#### SetLeafBlockByGroupView origin_group->ops.size(): "
          << origin_group->ops.size();
  for (auto op : origin_group->ops) {
    VLOG(1) << "##### op : " << op->name();
  }

  auto new_group = CloneGroup(origin_group, block, &ir_mapping);
  CHECK_EQ(origin_group->ops.size(), new_group->ops.size());

  // Insert YieldOp for outputs
  std::vector<pir::Value> outputs;
  builder.SetInsertionPointToBlockEnd(block);
  for (auto output : origin_group->GetGroupOutputValues()) {
    outputs.push_back(ir_mapping.Lookup(output));
    VLOG(1) << "##### output: " << pir::GetValueId(&output)
            << " new output: " << pir::GetValueId(&outputs.back());
  }
  VLOG(1) << "###### Insert YieldOp for outputs: " << outputs.size();
  builder.Build<pir::YieldOp>(outputs);

  UpdateShapeAnalysis(new_group,
                      ir_mapping,
                      value_dim_exprs_list,
                      value_to_dim_expr_idx,
                      new_group->shape_analysis.get());

  group_map->insert({block, new_group});
}

// Visit broadcast_tree by dfs
pir::Operation* CreateConditionBlock(
    const cinn::common::BroadcastTree& broadcast_tree,
    const cinn::dialect::ir::GroupPtr& origin_group,
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    const std::vector<pir::Value>& group_inputs,
    const std::vector<pir::Type>& output_types,
    pir::Builder& builder,  // NOLINT
    pir::Block* block,
    std::unordered_map<pir::Block*, cinn::dialect::ir::GroupPtr>* group_map) {
  if (broadcast_tree.Has<cinn::common::BroadcastLeaf>()) {
    const auto& broadcast_leaf =
        broadcast_tree.Get<cinn::common::BroadcastLeaf>();
    SetLeafBlockByGroupView(origin_group,
                            broadcast_leaf,
                            shape_analysis,
                            value_to_dim_expr_idx,
                            builder,
                            block,
                            group_map);
    return nullptr;
  } else {
    const auto& branch =
        broadcast_tree
            .Get<cinn::common::BroadcastBranch<cinn::common::BroadcastTree>>();
    const auto& [lhs_eq_rhs_cond, lhs_eq_one_cond, rhs_eq_one_cond] =
        BroadcastableToCondValue(
            branch.Get<0>(), shape_analysis, group_inputs, builder);

    // lhs == rhs
    auto lhs_eq_rhs_cond_op = builder.Build<paddle::dialect::IfOp>(
        lhs_eq_rhs_cond, std::vector<pir::Type>{output_types});
    pir::Block& lhs_eq_rhs_block = lhs_eq_rhs_cond_op.true_block();
    builder.SetInsertionPointToBlockEnd(&lhs_eq_rhs_block);
    CreateConditionBlock(branch.Get<1>(),
                         origin_group,
                         shape_analysis,
                         value_to_dim_expr_idx,
                         group_inputs,
                         output_types,
                         builder,
                         &lhs_eq_rhs_block,
                         group_map);

    pir::Block& lhs_not_eq_rhs_block = lhs_eq_rhs_cond_op.false_block();
    builder.SetInsertionPointToBlockEnd(&lhs_not_eq_rhs_block);

    // lhs != rhs && lhs == 1
    auto lhs_eq_one_cond_op = builder.Build<paddle::dialect::IfOp>(
        lhs_eq_one_cond, std::vector<pir::Type>{output_types});
    pir::Block& lhs_eq_one_block = lhs_eq_one_cond_op.true_block();
    builder.SetInsertionPointToBlockEnd(&lhs_eq_one_block);
    CreateConditionBlock(branch.Get<2>(),
                         origin_group,
                         shape_analysis,
                         value_to_dim_expr_idx,
                         group_inputs,
                         output_types,
                         builder,
                         &lhs_eq_one_block,
                         group_map);

    // lhs != rhs && rhs == 1
    pir::Block& rhs_eq_one_block = lhs_eq_one_cond_op.false_block();
    builder.SetInsertionPointToBlockEnd(&rhs_eq_one_block);
    CreateConditionBlock(branch.Get<3>(),
                         origin_group,
                         shape_analysis,
                         value_to_dim_expr_idx,
                         group_inputs,
                         output_types,
                         builder,
                         &rhs_eq_one_block,
                         group_map);

    return lhs_eq_rhs_cond_op;
  }
}

std::unordered_map<cinn::dialect::ir::GroupPtr,
                   std::unordered_map<std::string, pir::Attribute>>
ComplieGroupAsOpAttribute(
    const std::shared_ptr<cinn::hlir::framework::PirCompiler>& pir_compiler,
    const std::unordered_map<pir::Block*, cinn::dialect::ir::GroupPtr>&
        group_map) {
  std::vector<cinn::dialect::ir::GroupPtr> group_list;
  std::unordered_map<cinn::dialect::ir::GroupPtr, size_t> group_to_idx;
  for (auto& [_, group] : group_map) {
    group_list.push_back(group);
    group_to_idx[group] = group_list.size() - 1;
  }

  auto fn_ptr_res = pir_compiler->BuildCUDAJITInfo(group_list);

  std::unordered_map<cinn::dialect::ir::GroupPtr,
                     std::unordered_map<std::string, pir::Attribute>>
      result;
  for (auto& [__acosf64, group] : group_map) {
    std::unordered_map<std::string, ::pir::Attribute> op_attrs{
        {cinn::dialect::JitKernelOp::kAttrName,
         cinn::dialect::CINNKernelInfoAttribute::get(
             pir::IrContext::Instance(), fn_ptr_res[group_to_idx.at(group)])},
    };
    result.insert({group, op_attrs});
  }
  return result;
}

pir::Operation* ProcessGroup(
    const cinn::dialect::ir::GroupPtr& group,
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis,
    const std::shared_ptr<cinn::hlir::framework::PirCompiler>& pir_compiler,
    const std::unordered_map<pir::Value, pir::Value>& value_map,
    pir::PatternRewriter& rewriter) {  // NOLINT
  std::vector<pir::Operation*> group_ops_view;
  std::unordered_set<pir::Value> value_view;
  group->WalkOps([&group, &group_ops_view, &value_view](pir::Operation* op) {
    group_ops_view.push_back(op);
    VLOG(1) << "####### group@" << group.get() << " : " << op->name() << " @"
            << op;
    for (size_t i = 0; i < op->num_operands(); ++i) {
      value_view.insert(op->operand_source(i));
    }
    for (size_t i = 0; i < op->num_results(); ++i) {
      value_view.insert(op->result(i));
      auto val = op->result(i);
    }
  });

  // 1. construct broadcast tree
  VLOG(1) << "construct broadcast tree";
  cinn::adt::List<std::vector<symbol::DimExpr>> all_value_dim_exprs;
  std::unordered_map<pir::Value, size_t> value_to_dim_expr_idx;
  for (auto value : value_view) {
    const auto& shape_dim_expr = shape_analysis->GetShapeOrDataForValue(&value);
    const auto& data_shape = shape_dim_expr.data();
    VLOG(1) << "#### value : " << pir::GetValueId(&value) << " : "
            << shape_dim_expr;
    if (data_shape) {
      all_value_dim_exprs->push_back(*data_shape);
    } else {
      all_value_dim_exprs->push_back(shape_dim_expr.shape());
    }
    value_to_dim_expr_idx[value] = all_value_dim_exprs->size() - 1;
  }
  cinn::common::BroadcastTree broadcast_tree =
      cinn::common::ConstructBroadcastTree(
          cinn::common::BroadcastLeaf(all_value_dim_exprs));

  // 2. broadcast tree to condition op
  VLOG(1) << "broadcast tree to condition op";
  auto group_inputs = GetBlockOutsideInput(group->ops);
  for (size_t i = 0; i < group_inputs.size(); ++i) {
    if (value_map.find(group_inputs[i]) != value_map.end()) {
      shape_analysis->SetShapeOrDataForValue(
          &value_map.at(group_inputs[i]),
          shape_analysis->GetShapeOrDataForValue(&group_inputs[i]));
      group_inputs[i] = value_map.at(group_inputs[i]);
    }
  }

  std::vector<pir::Type> output_types;
  auto group_output_values = group->GetGroupOutputValues();
  for (size_t i = 0; i < group_output_values.size(); ++i) {
    output_types.push_back(group_output_values[i].type());
  }

  auto* origin_block = rewriter.block();

  std::unordered_map<pir::Block*, cinn::dialect::ir::GroupPtr> group_map{
      {origin_block, group}};
  pir::Operation* cond_op = CreateConditionBlock(broadcast_tree,
                                                 group,
                                                 shape_analysis,
                                                 value_to_dim_expr_idx,
                                                 group_inputs,
                                                 output_types,
                                                 rewriter,
                                                 rewriter.block(),
                                                 &group_map);

  if (cond_op) {
    group_map.erase(origin_block);
  }
  auto* program = origin_block->parent_program();
  VLOG(4) << "Before simply condition block: " << *program;
  shape_analysis->PrintAllShapeOrDataDimExprs();
  // 3. simply every condition block
  VLOG(1) << "simply condition block";
  for (auto& [block, group] : group_map) {
    EraseExpandsInBlock(block,
                        rewriter,
                        [&shape_analysis](pir::Value value)
                            -> const symbol::ShapeOrDataDimExprs& {
                          return shape_analysis->GetShapeOrDataForValue(&value);
                        });
    ReplaceExpandWithBroadcast(
        rewriter.ir_context(), block, group->shape_analysis);

    std::vector<pir::Operation*> group_new_ops;
    group_new_ops.reserve(block->size());
    std::unordered_set<pir::Operation*> group_ops_set;
    for (auto& op : *block) {
      if (!op.isa<pir::YieldOp>()) {
        group_new_ops.push_back(&op);
        group_ops_set.insert(&op);
      }
    }
    group->ops = group_new_ops;
    group->ops_set = group_ops_set;
  }
  VLOG(4) << "After simply condition block: " << *program;

  // 4. complie condition block to jit_kernel_op
  VLOG(1) << "complie condition block to jit_kernel_op";
  // prepare attribute for jit_kernel_op
  auto op_attr_map = ComplieGroupAsOpAttribute(pir_compiler, group_map);
  // create jit_kernel_op
  if (cond_op) {  // has condition block
    for (auto& [block, group] : group_map) {
      rewriter.SetInsertionPointToBlockEnd(block);
      auto jit_kernel_op = rewriter.Build<cinn::dialect::JitKernelOp>(
          group_inputs, op_attr_map.at(group), output_types);
      auto group_output_values = group->GetGroupOutputValues();
      CHECK(jit_kernel_op.num_results() == group_output_values.size());
      for (size_t i = 0; i < jit_kernel_op.num_results(); ++i) {
        rewriter.ReplaceAllUsesWith(group_output_values[i],
                                    jit_kernel_op.result(i));
      }
    }
    return cond_op;
  } else {  // no condition block
    CHECK_EQ(group_map.size(), 1UL);
    auto jit_kernel_op = rewriter.Build<cinn::dialect::JitKernelOp>(
        group_inputs, op_attr_map.at(group_map.begin()->second), output_types);
    return jit_kernel_op;
  }
}

class GroupOpPattern : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
 public:
  GroupOpPattern(
      ::pir::IrContext* context,
      const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis)
      : pir::OpRewritePattern<cinn::dialect::GroupOp>(context),
        shape_analysis_(shape_analysis) {}

  bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto target = cinn::common::DefaultNVGPUTarget();
    auto* program = group_op->GetParentProgram();
    VLOG(4) << "Before GroupOpPattern: " << *program;
    // TODO(Aurelius84): Remove scope after cleaning PirCompiler usless Build
    // Interface
    auto scope = std::make_shared<cinn::hlir::framework::Scope>();

    VLOG(4) << "start Lowering Group Op: " << group_op;
    // using yield op to sort
    std::unordered_map<::pir::Value, size_t> value2id;
    auto yeild_op = group_op.ops().back();
    for (size_t i = 0; i < yeild_op->num_operands(); ++i) {
      value2id[yeild_op->operand_source(i)] = i;
    }
    std::unordered_map<pir::Value, pir::Value> value_map;
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(group_op->GetParentProgram());
    auto shape_analysis_ =
        std::make_shared<pir::ShapeConstraintIRAnalysis>(shape_analysis);

    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(group_op->GetParentProgram());
    shape_analysis.PrintAllShapeOrDataDimExprs();
    auto shape_analysis_ =
        std::make_shared<pir::ShapeConstraintIRAnalysis>(shape_analysis);
    VLOG(1) << "shape_analysis: " << &shape_analysis
            << " program:" << group_op->GetParentProgram();
    shape_analysis.PrintAllShapeOrDataDimExprs();

    // op fusion
    auto op_fusion = cinn::dialect::ir::OpFusionPassInternal(
        GetOpListNotIncludeYield(group_op.ops()),
        GetOutputOpList(group_op.ops()),
        shape_analysis_);

    // fusion merge
    auto group_list = cinn::dialect::ir::GeneralFusionMergePassInternal(
        op_fusion, shape_analysis_);
    VLOG(1) << "###### GeneralFusionMergePass op_fusion size: "
            << op_fusion.size();

    for (auto group : group_list) {
      auto ir_compiler = cinn::hlir::framework::PirCompilerManager::Create(
          *program, target, scope);
      group->shape_analysis = shape_analysis_;
      if (FLAGS_cinn_enable_map_expr) {
        cinn::adt::TryGenerateMapExprFromGroup(group);
      }

      pir::Operation* complied_op = ProcessGroup(
          group, shape_analysis_, ir_compiler, value_map, rewriter);
      auto group_output_values = group->GetGroupOutputValues();
      for (size_t i = 0; i < complied_op->num_results(); ++i) {
        auto find_it = value2id.find(group_output_values[i]);
        if (find_it != value2id.end()) {
          rewriter.ReplaceAllUsesWith(group_op.result(find_it->second),
                                      complied_op->result(i));
        }
        value_map[group_output_values[i]] = complied_op->result(i);
      }
    }
    value_map.clear();
    VLOG(4) << "Before GroupOpPattern.EraseOp: " << *program;
    rewriter.EraseOp(group_op);
    return true;
  }

 private:
  std::shared_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis_{nullptr};
};

class CinnGroupLoweringPass : public pir::PatternRewritePass {
 public:
  CinnGroupLoweringPass(
      const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis)
      : pir::PatternRewritePass("cinn_group_lowering", 1),
        shape_analysis_(shape_analysis) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<GroupOpPattern>(context, shape_analysis_);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  std::shared_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis_{nullptr};
};

}  // namespace

namespace cinn {
namespace dialect {
namespace ir {

std::unique_ptr<::pir::Pass> CreateCinnGroupLoweringPass(
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  return std::make_unique<CinnGroupLoweringPass>(shape_analysis);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

// REGISTER_IR_PASS(cinn_group_lowering, CinnGroupLoweringPass);
