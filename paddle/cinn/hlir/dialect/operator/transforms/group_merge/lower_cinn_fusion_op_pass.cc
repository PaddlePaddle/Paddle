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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/lower_cinn_fusion_op_pass.h"

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
#include "paddle/cinn/hlir/framework/pir/utils.h"
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

using cinn::dialect::ir::Group;
using cinn::hlir::framework::pir::CompatibleInfo;

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

void EraseUneccessaryExpandsInBlock(
    pir::Block* block,
    pir::PatternRewriter& rewriter,  // NOLINT
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  while (EraseOneExpand(block, rewriter, ShapeOrDataDimExprs4Value)) {
    // Do nothing.
  }
}

void ReplaceExpandWithBroadcast(pir::IrContext* ir_context,
                                pir::Block* block,
                                const cinn::dialect::ir::GroupPtr& group) {
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
      auto x_rank = op->operand_source(0)
                        .type()
                        .dyn_cast<pir::ShapedTypeInterface>()
                        .GetRank();
      auto out_rank =
          op->result(0).type().dyn_cast<pir::ShapedTypeInterface>().GetRank();
      std::vector<int64_t> broadcast_axes(x_rank, 0);
      size_t index_gap = out_rank - x_rank;
      for (size_t i = 0; i < x_rank; ++i) {
        broadcast_axes[i] = i + index_gap;
      }
      std::vector<int64_t> out_shape(out_rank, -1);
      auto broadcast = builder.Build<cinn::dialect::BroadcastOp>(
          op->operand_source(0), broadcast_axes, out_shape);
      auto broadcast_out = broadcast.result(0);
      auto expand_out = op->result(0);
      expand_out.ReplaceAllUsesWith(broadcast_out);
      group->value_to_shape_or_data_exprs.emplace(
          broadcast_out, group->GetShapeOrDataExprs(expand_out));
      CHECK(op->use_empty());
      auto generate_shape_op = op->operand_source(1).defining_op();
      op->Erase();
      generate_shape_op->Erase();
    }
  }
}

std::vector<pir::Value> GetBlockOutsideInput(
    const std::vector<pir::Operation*>& op_list) {
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

std::tuple<pir::Value, pir::Value, pir::Value> BroadcastableToCondValue(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    pir::ShapeConstraintIRAnalysis& shape_analysis,  // NOLINT
    const std::vector<pir::Value>& group_inputs,
    pir::Builder& builder) {  // NOLINT
  const auto& lhs_expr = broadcastable_condition->lhs;
  const auto& rhs_expr = broadcastable_condition->rhs;
  auto ShapeOrDataDimExprs4Value = [&shape_analysis](pir::Value value) {
    return shape_analysis.GetShapeOrDataForValue(value);
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
  return group->Clone(block, *ir_mapping);
}

void UpdateGroupShapeExprs(
    const cinn::dialect::ir::GroupPtr& new_group,
    const cinn::dialect::ir::GroupPtr& origin_group,
    const pir::IrMapping& ir_mapping,
    const cinn::common::BroadcastLeaf& value_dim_exprs_list,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx) {
  for (const auto& [origin_val, new_val] : ir_mapping.GetMap<pir::Value>()) {
    const auto& shape_dim_expr =
        value_dim_exprs_list->at(value_to_dim_expr_idx.at(origin_val));
    const auto& origin_shape_or_data =
        origin_group->GetShapeOrDataExprs(origin_val);
    if (origin_shape_or_data.data()) {
      new_group->value_to_shape_or_data_exprs.emplace(
          new_val,
          symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
              std::vector<symbol::DimExpr>{shape_dim_expr.size()},
              shape_dim_expr)});
    } else {
      new_group->value_to_shape_or_data_exprs.emplace(
          new_val,
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs(shape_dim_expr)});
    }
  }
}

void SetLeafBlockByGroupView(
    const cinn::dialect::ir::GroupPtr& origin_group,
    const cinn::common::BroadcastLeaf& value_dim_exprs_list,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    pir::Builder& builder,  // NOLINT
    pir::Block* block,
    std::unordered_map<pir::Block*, cinn::dialect::ir::GroupPtr>* group_map) {
  pir::IrMapping ir_mapping;
  auto origin_group_inputs = GetBlockOutsideInput(origin_group->ops);
  for (auto input : origin_group_inputs) {
    ir_mapping.Add(input, input);
  }

  auto new_group = CloneGroup(origin_group, block, &ir_mapping);
  CHECK_EQ(origin_group->ops.size(), new_group->ops.size());
  UpdateGroupShapeExprs(new_group,
                        origin_group,
                        ir_mapping,
                        value_dim_exprs_list,
                        value_to_dim_expr_idx);

  // Insert YieldOp for outputs
  std::vector<pir::Value> outputs;
  builder.SetInsertionPointToBlockEnd(block);
  for (auto output : origin_group->GetGroupOutputValues()) {
    outputs.push_back(ir_mapping.Lookup(output));
  }
  builder.Build<pir::YieldOp>(outputs);

  group_map->insert({block, new_group});
}

std::vector<pir::Value> GetOpOuputValues(const pir::Operation* op) {
  std::vector<pir::Value> outputs;
  outputs.reserve(op->num_results());
  for (size_t i = 0; i < op->num_results(); ++i) {
    outputs.push_back(op->result(i));
  }
  return outputs;
}

void InsertYieldOpForCondBlock(pir::Operation* cond_op,
                               pir::Builder& builder) {  // NOLINT
  if (cond_op) {
    builder.SetInsertionPointAfter(cond_op);
    builder.Build<pir::YieldOp>(GetOpOuputValues(cond_op));
  }
}

// Visit broadcast_tree by dfs
pir::Operation* CreateConditionBlock(
    const cinn::common::BroadcastTree& broadcast_tree,
    const cinn::dialect::ir::GroupPtr& origin_group,
    pir::ShapeConstraintIRAnalysis& shape_analysis,  // NOLINT
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
    auto* lhs_eq_rhs_block_op = CreateConditionBlock(branch.Get<1>(),
                                                     origin_group,
                                                     shape_analysis,
                                                     value_to_dim_expr_idx,
                                                     group_inputs,
                                                     output_types,
                                                     builder,
                                                     &lhs_eq_rhs_block,
                                                     group_map);
    InsertYieldOpForCondBlock(lhs_eq_rhs_block_op, builder);

    pir::Block& lhs_not_eq_rhs_block = lhs_eq_rhs_cond_op.false_block();
    builder.SetInsertionPointToBlockEnd(&lhs_not_eq_rhs_block);

    // lhs != rhs && lhs == 1
    auto lhs_eq_one_cond_op = builder.Build<paddle::dialect::IfOp>(
        lhs_eq_one_cond, std::vector<pir::Type>{output_types});
    pir::Block& lhs_eq_one_block = lhs_eq_one_cond_op.true_block();
    builder.SetInsertionPointToBlockEnd(&lhs_eq_one_block);
    auto* lhs_eq_one_block_op = CreateConditionBlock(branch.Get<2>(),
                                                     origin_group,
                                                     shape_analysis,
                                                     value_to_dim_expr_idx,
                                                     group_inputs,
                                                     output_types,
                                                     builder,
                                                     &lhs_eq_one_block,
                                                     group_map);
    InsertYieldOpForCondBlock(lhs_eq_one_block_op, builder);

    // lhs != rhs && rhs == 1
    pir::Block& rhs_eq_one_block = lhs_eq_one_cond_op.false_block();
    builder.SetInsertionPointToBlockEnd(&rhs_eq_one_block);
    auto* rhs_eq_one_block_op = CreateConditionBlock(branch.Get<3>(),
                                                     origin_group,
                                                     shape_analysis,
                                                     value_to_dim_expr_idx,
                                                     group_inputs,
                                                     output_types,
                                                     builder,
                                                     &rhs_eq_one_block,
                                                     group_map);
    InsertYieldOpForCondBlock(rhs_eq_one_block_op, builder);

    builder.SetInsertionPointToBlockEnd(&lhs_not_eq_rhs_block);
    builder.Build<pir::YieldOp>(GetOpOuputValues(lhs_eq_one_cond_op));

    return lhs_eq_rhs_cond_op;
  }
}

std::unordered_map<cinn::dialect::ir::GroupPtr,
                   std::unordered_map<std::string, pir::Attribute>>
ComplieGroupAsOpAttribute(
    const std::shared_ptr<cinn::hlir::framework::PirCompiler>& pir_compiler,
    const std::vector<cinn::dialect::ir::GroupPtr>& group_list) {
  auto fn_ptr_res = pir_compiler->BuildCUDAJITInfo(group_list);

  std::unordered_map<cinn::dialect::ir::GroupPtr,
                     std::unordered_map<std::string, pir::Attribute>>
      result;
  for (size_t i = 0; i < group_list.size(); ++i) {
    std::unordered_map<std::string, ::pir::Attribute> op_attrs{
        {cinn::dialect::JitKernelOp::kAttrName,
         cinn::dialect::CINNKernelInfoAttribute::get(pir::IrContext::Instance(),
                                                     fn_ptr_res[i])},
    };
    result.insert({group_list[i], op_attrs});
  }
  return result;
}

void SimplyConditionBlock(
    pir::PatternRewriter& rewriter,  // NOLINT
    std::unordered_map<pir::Block*, cinn::dialect::ir::GroupPtr>* group_map) {
  VLOG(4) << "simply condition block";
  using DoEachMutBlockGroupT =
      std::function<void(pir::Block*, const cinn::dialect::ir::GroupPtr&)>;
  const auto& ForEachMutBlockGroup = [&](const DoEachMutBlockGroupT& DoEach) {
    for (auto& [block, group] : *group_map) {
      DoEach(block, group);
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
  };
  ForEachMutBlockGroup([&](auto* block, const auto& group) {
    auto GetShapeOrDataForValue =
        [&group](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
      return group->GetShapeOrDataExprs(value);
    };
    EraseUneccessaryExpandsInBlock(block, rewriter, GetShapeOrDataForValue);
  });
  ForEachMutBlockGroup([&](auto* block, const auto& group) {
    ReplaceExpandWithBroadcast(rewriter.ir_context(), block, group);
  });
}

void CompileGroupToJitKernelOp(
    const std::vector<pir::Value>& group_inputs,
    const std::vector<pir::Type>& output_types,
    const std::shared_ptr<cinn::hlir::framework::PirCompiler>& pir_compiler,
    pir::PatternRewriter& rewriter,  // NOLINT
    std::unordered_map<pir::Block*, cinn::dialect::ir::GroupPtr>* group_map) {
  // prepare attribute for jit_kernel_op
  std::vector<cinn::dialect::ir::GroupPtr> group_list;
  group_list.reserve(group_map->size());
  for (const auto& [_, group] : *group_map) {
    group_list.push_back(group);
  }
  auto op_attr_map = ComplieGroupAsOpAttribute(pir_compiler, group_list);
  VLOG(4) << "The size of group_map is : " << group_map->size();
  for (auto& [block, group] : *group_map) {
    auto& yeild_op = block->back();
    CHECK(yeild_op.isa<pir::YieldOp>()) << "Last op of block should be yield";
    rewriter.set_insertion_point(&yeild_op);
    auto jit_kernel_op = rewriter.Build<cinn::dialect::JitKernelOp>(
        group_inputs, op_attr_map.at(group), output_types);
    auto group_output_values = group->GetGroupOutputValues();
    CHECK(jit_kernel_op.num_results() == group_output_values.size());
    for (size_t i = 0; i < jit_kernel_op.num_results(); ++i) {
      rewriter.ReplaceAllUsesWith(group_output_values[i],
                                  jit_kernel_op.result(i));
    }

    // Delete origin group ops
    std::vector<pir::Operation*> group_ops;
    for (auto iter = block->rbegin(); iter != block->rend(); iter++) {
      if (!iter->isa<pir::YieldOp>()) {
        group_ops.push_back(&(*iter));
      }
    }
    for (auto* op : group_ops) {
      if (op->use_empty()) {
        op->Erase();
      }
    }
  }
}

pir::Operation* ComplieBroadcastTreeToConditionBlock(
    const cinn::common::BroadcastTree& broadcast_tree,
    const cinn::dialect::ir::GroupPtr& group,
    pir::ShapeConstraintIRAnalysis& shape_analysis,  // NOLINT
    const std::shared_ptr<cinn::hlir::framework::PirCompiler>& pir_compiler,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    const std::vector<pir::Value>& group_inputs,
    const std::vector<pir::Type>& output_types,
    pir::PatternRewriter& rewriter) {  // NOLINT
  // 1. broadcast tree to condition op
  VLOG(4) << "broadcast tree to condition op";
  std::unordered_map<pir::Block*, cinn::dialect::ir::GroupPtr> group_map;
  pir::Operation* cond_op = CreateConditionBlock(broadcast_tree,
                                                 group,
                                                 shape_analysis,
                                                 value_to_dim_expr_idx,
                                                 group_inputs,
                                                 output_types,
                                                 rewriter,
                                                 rewriter.block(),
                                                 &group_map);
  // 2. simply every condition block
  auto* program = group->ops.front()->GetParentProgram();
  VLOG(6) << "Before simply condition block: " << *program;

  SimplyConditionBlock(rewriter, &group_map);
  VLOG(6) << "After simply condition block: " << *program;

  // 3. complie condition block to jit_kernel_op
  CompileGroupToJitKernelOp(
      group_inputs, output_types, pir_compiler, rewriter, &group_map);
  VLOG(6) << "complie condition block to jit_kernel_op: " << *program;

  return cond_op;
}

pir::Operation* ProcessDyShapeGroup(
    const cinn::dialect::ir::GroupPtr& group,
    pir::ShapeConstraintIRAnalysis& shape_analysis,  // NOLINT
    const std::shared_ptr<cinn::hlir::framework::PirCompiler>& pir_compiler,
    pir::PatternRewriter& rewriter) {  // NOLINT
  std::unordered_set<pir::Value> value_view;
  group->WalkOps([&group, &value_view](pir::Operation* op) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      value_view.insert(op->operand_source(i));
    }
    for (size_t i = 0; i < op->num_results(); ++i) {
      value_view.insert(op->result(i));
    }
  });

  // construct broadcast tree
  VLOG(4) << "construct broadcast tree";
  cinn::adt::List<std::vector<symbol::DimExpr>> all_value_dim_exprs;
  std::unordered_map<pir::Value, size_t> value_to_dim_expr_idx;
  for (auto value : value_view) {
    const auto& shape_dim_expr = group->GetShapeOrDataExprs(value);
    const auto& data_shape = shape_dim_expr.data();
    if (data_shape) {
      all_value_dim_exprs->push_back(*data_shape);
    } else {
      all_value_dim_exprs->push_back(shape_dim_expr.shape());
    }
    value_to_dim_expr_idx[value] = all_value_dim_exprs->size() - 1;
  }
  VLOG(6) << "before constructed. broadcast-leaf: \n"
          << ToTxtString(cinn::common::BroadcastTree(all_value_dim_exprs));
  cinn::common::BroadcastTree broadcast_tree =
      cinn::common::ConstructBroadcastTree(
          cinn::common::BroadcastLeaf(all_value_dim_exprs));
  VLOG(4) << "broadcast-tree: \n" << ToTxtString(broadcast_tree);

  auto group_inputs = GetBlockOutsideInput(group->ops);

  // has multiple branch
  if (broadcast_tree
          .Has<cinn::common::BroadcastBranch<cinn::common::BroadcastTree>>()) {
    std::vector<pir::Type> output_types;
    auto group_output_values = group->GetGroupOutputValues();
    for (size_t i = 0; i < group_output_values.size(); ++i) {
      output_types.push_back(group_output_values[i].type());
    }
    return ComplieBroadcastTreeToConditionBlock(broadcast_tree,
                                                group,
                                                shape_analysis,
                                                pir_compiler,
                                                value_to_dim_expr_idx,
                                                group_inputs,
                                                output_types,
                                                rewriter);
  } else {  // no condition block
    // complie group to jit_kernel_op
    auto op_attr_map = ComplieGroupAsOpAttribute(pir_compiler, {group});
    std::vector<pir::Type> output_types;
    const auto& group_output_values = group->output_values;
    for (size_t i = 0; i < group_output_values.size(); ++i) {
      output_types.push_back(group_output_values[i].type());
    }
    auto jit_kernel_op = rewriter.Build<cinn::dialect::JitKernelOp>(
        group_inputs, op_attr_map.at(group), output_types);
    return jit_kernel_op;
  }
}

std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>
CreateGroupShapeOrDataExprs(
    const cinn::dialect::ir::GroupPtr& group,
    pir::ShapeConstraintIRAnalysis& shape_analysis) {  // NOLINT
  std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs> value2shape;
  for (auto* op : group->ops) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      auto operand = op->operand_source(i);
      if (shape_analysis.HasShapeOrDataForValue(operand)) {
        value2shape.insert(
            {operand, shape_analysis.GetShapeOrDataForValue(operand)});
      }
    }
    for (size_t i = 0; i < op->num_results(); ++i) {
      auto result = op->result(i);
      if (value2shape.find(result) == value2shape.end() &&
          shape_analysis.HasShapeOrDataForValue(result)) {
        value2shape.insert(
            {result, shape_analysis.GetShapeOrDataForValue(result)});
      }
    }
  }
  return value2shape;
}

class FusionOpPattern : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  explicit FusionOpPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::FusionOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto target = cinn::common::DefaultNVGPUTarget();
    // TODO(Aurelius84): Remove scope after cleaning PirCompiler usless Build
    // Interface
    auto scope = std::make_shared<cinn::hlir::framework::Scope>();
    auto* program = fusion_op->GetParentProgram();
    auto ir_compiler = cinn::hlir::framework::PirCompilerManager::Create(
        *program, target, scope);
    auto group = RebuildGroup(fusion_op);
    // Because the group is rebuilt, the order of group.output_values generated
    // by BuildCUDAJITInfo may not be same with the order bound in the yield op,
    // so a mapping is required.
    std::unordered_map<::pir::Value, size_t> value2id;

    auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(
        fusion_op->GetParentProgram());
    group->value_to_shape_or_data_exprs =
        CreateGroupShapeOrDataExprs(group, shape_analysis);
    if (FLAGS_cinn_enable_map_expr) {
      cinn::adt::TryGenerateMapExprFromGroup(group);
    }

    // TODO(zhangyuqin1998): Replace pir::Group with a new structure
    pir::Operation* complied_op =
        ProcessGroup(group, shape_analysis, ir_compiler, rewriter);

    // the output_values of group may be changed.
    for (size_t i = 0; i < group->output_values.size(); ++i) {
      value2id[group->output_values[i]] = i;
    }

    auto yeild_op = fusion_op.GetOperators().back();
    for (size_t i = 0; i < fusion_op.num_results(); ++i) {
      rewriter.ReplaceAllUsesWith(
          fusion_op.result(i),
          complied_op->result(value2id[yeild_op->operand_source(i)]));
      if (shape_analysis.HasShapeOrDataForValue(fusion_op.result(i))) {
        shape_analysis.SetShapeOrDataForValue(
            complied_op->result(value2id[yeild_op->operand_source(i)]),
            shape_analysis.GetShapeOrDataForValue(fusion_op.result(i)));
      } else {
        LOG(WARNING) << "No shape_data for "
                     << fusion_op.result(i).defining_op()->name() << "_result_"
                     << i;
      }
    }

    rewriter.EraseOp(fusion_op);
    return true;
  }

 protected:
  virtual pir::Operation* ProcessGroup(
      const cinn::dialect::ir::GroupPtr& group,
      pir::ShapeConstraintIRAnalysis& shape_analysis,  // NOLINT
      const std::shared_ptr<cinn::hlir::framework::PirCompiler>& pir_compiler,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    auto group_inputs = GetBlockOutsideInput(group->ops);
    // complie group to jit_kernel_op
    auto op_attr_map = ComplieGroupAsOpAttribute(pir_compiler, {group});
    std::vector<pir::Type> output_types;
    const auto& group_output_values = group->output_values;
    for (size_t i = 0; i < group_output_values.size(); ++i) {
      output_types.push_back(group_output_values[i].type());
    }
    auto jit_kernel_op = rewriter.Build<cinn::dialect::JitKernelOp>(
        group_inputs, op_attr_map.at(group), output_types);
    return jit_kernel_op;
  }

 private:
  std::shared_ptr<Group> RebuildGroup(cinn::dialect::FusionOp fusion_op) const {
    auto group = std::make_shared<Group>();
    group->op_pattern_kind = cinn::hlir::framework::OpPatternKind::kElementWise;

    // Rebuild ops of the group
    for (auto op : fusion_op.GetOperators()) {
      if (!op->isa<::pir::YieldOp>()) {
        group->ops.push_back(op);
        group->ops_set.insert(op);
        group->op_pattern_kind =
            static_cast<int>(CompatibleInfo::OpKind(*op)) >
                    static_cast<int>(group->op_pattern_kind)
                ? CompatibleInfo::OpKind(*op)
                : group->op_pattern_kind;
      }
    }

    // Rebuild output_ops and input_ops of the group
    auto yeild_op = fusion_op.GetOperators().back();
    for (size_t i = 0; i < yeild_op->num_operands(); ++i) {
      group->output_ops.insert(yeild_op->operand_source(i).defining_op());
    }

    // Rebuild other informations
    // TODO(zhangyuqin1998): Do we need group.master_ops?
    return group;
  }
};

class DyShapeFusionOpPattern : public FusionOpPattern {
 public:
  using FusionOpPattern::FusionOpPattern;

 protected:
  virtual pir::Operation* ProcessGroup(
      const cinn::dialect::ir::GroupPtr& group,
      pir::ShapeConstraintIRAnalysis& shape_analysis,  // NOLINT
      const std::shared_ptr<cinn::hlir::framework::PirCompiler>& pir_compiler,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    return ProcessDyShapeGroup(group, shape_analysis, pir_compiler, rewriter);
  }
};

class LowerCinnFusionOpPass : public pir::PatternRewritePass {
 public:
  LowerCinnFusionOpPass()
      : pir::PatternRewritePass("lower_cinn_fusion_op", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<FusionOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

class LowerCinnDyShapeFusionOpPass : public pir::PatternRewritePass {
 public:
  LowerCinnDyShapeFusionOpPass()
      : pir::PatternRewritePass("lower_cinn_dynamic_shape_fusion_op", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<DyShapeFusionOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

}  // namespace

namespace cinn {
namespace dialect {
namespace ir {

std::unique_ptr<::pir::Pass> CreateLowerCinnFusionOpPass() {
  return std::make_unique<LowerCinnFusionOpPass>();
}

std::unique_ptr<::pir::Pass> CreateLowerCinnDyShapeFusionOpPass() {
  return std::make_unique<LowerCinnDyShapeFusionOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

// REGISTER_IR_PASS(cinn_group_lowering, LowerCinnFusionOpPass);
