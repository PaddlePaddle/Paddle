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

#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/broadcast_with_cf.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"

using OpLoweringGroup = cinn::hlir::framework::pir::OpLoweringGroup;
using OpLoweringGroupPtr = std::shared_ptr<OpLoweringGroup>;
using cinn::dialect::ir::details::CompileGroupAsOpAttribute;
using cinn::dialect::ir::details::GetBlockOutsideInput;

namespace {
std::vector<pir::Value> GetOpOuputValues(const pir::Operation* op) {
  std::vector<pir::Value> outputs;
  outputs.reserve(op->num_results());
  for (size_t i = 0; i < op->num_results(); ++i) {
    outputs.push_back(op->result(i));
  }
  return outputs;
}

using ShapeOrDataDimExprs4ValueT =
    std::function<const symbol::ShapeOrDataDimExprs&(pir::Value)>;

static bool SameInputOutputShape(
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

void CompileGroupToJitKernelOp(
    pir::PatternRewriter& rewriter,  // NOLINT
    std::unordered_map<pir::Block*, OpLoweringGroupPtr>* group_map) {
  // prepare attribute for jit_kernel_op
  std::vector<OpLoweringGroupPtr> group_list;
  group_list.reserve(group_map->size());
  for (const auto& [_, group] : *group_map) {
    group_list.push_back(group);
  }
  auto op_attr_map = CompileGroupAsOpAttribute(group_list);
  VLOG(4) << "The size of group_map is : " << group_map->size();
  for (auto& [block, group] : *group_map) {
    auto& yield_op = block->back();
    CHECK(yield_op.isa<pir::YieldOp>()) << "Last op of block should be yield";
    std::vector<pir::Type> output_types;
    const auto& group_output_values = yield_op.operands_source();
    for (size_t i = 0; i < group_output_values.size(); ++i) {
      output_types.push_back(group_output_values[i].type());
    }
    rewriter.set_insertion_point(&yield_op);
    const auto& group_inputs = GetBlockOutsideInput(group->ops());
    auto jit_kernel_op = rewriter.Build<cinn::dialect::JitKernelOp>(
        group_inputs, op_attr_map.at(group), output_types);
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

void UpdateGroupShapeExprs(
    const OpLoweringGroupPtr& new_group,
    const OpLoweringGroupPtr& origin_group,
    const pir::IrMapping& ir_mapping,
    const cinn::common::BroadcastLeaf& value_dim_exprs_list,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx) {
  for (const auto& [origin_val, new_val] : ir_mapping.GetMap<pir::Value>()) {
    const auto& shape_dim_expr =
        value_dim_exprs_list->at(value_to_dim_expr_idx.at(origin_val));
    const auto& origin_shape_or_data =
        origin_group->GetShapeOrDataExprs(origin_val);
    if (origin_shape_or_data.data()) {
      std::vector<symbol::DimExpr> shape_dim_expr_shape = {
          symbol::DimExpr(static_cast<int64_t>(shape_dim_expr.size()))};
      new_group->SetShapeOrDataExprs(
          new_val,
          symbol::ShapeOrDataDimExprs{symbol::TensorShapeOrDataDimExprs(
              shape_dim_expr_shape, shape_dim_expr)});
    } else {
      new_group->SetShapeOrDataExprs(
          new_val,
          symbol::ShapeOrDataDimExprs{
              symbol::TensorShapeOrDataDimExprs(shape_dim_expr)});
    }
  }
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
    PADDLE_ENFORCE_NOT_NULL(generate_shape_op,
                            phi::errors::PreconditionNotMet(
                                "The generate shape op must not be null."));
    rewriter.ReplaceAllUsesWith(expand.out(), expand.x());
    rewriter.EraseOp(expand);
    if (generate_shape_op->use_empty()) {
      rewriter.EraseOp(generate_shape_op);
    }
    return true;
  }
  return false;
}

void EraseUnnecessaryExpandsInBlock(
    pir::Block* block,
    pir::PatternRewriter& rewriter,  // NOLINT
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  while (EraseOneExpand(block, rewriter, ShapeOrDataDimExprs4Value)) {
    // Do nothing.
  }
}

void ReplaceExpandWithBroadcast(pir::IrContext* ir_context,
                                pir::Block* block,
                                const OpLoweringGroupPtr& group) {
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
      group->SetShapeOrDataExprs(broadcast_out,
                                 group->GetShapeOrDataExprs(expand_out));
      CHECK(op->use_empty());
      auto generate_shape_op = op->operand_source(1).defining_op();
      op->Erase();
      if (generate_shape_op->use_empty()) {
        generate_shape_op->Erase();
      }
    }
  }
}

std::tuple<pir::Value, pir::Value, pir::Value> BroadcastableToCondValue(
    const symbol::Broadcastable<symbol::DimExpr>& broadcastable_condition,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
    const std::vector<pir::Value>& group_inputs,
    pir::Builder& builder) {  // NOLINT
  const auto& lhs_expr = broadcastable_condition->lhs;
  const auto& rhs_expr = broadcastable_condition->rhs;

  std::vector<pir::Value> lhs_minimal_inputs;
  std::vector<pir::Attribute> lhs_output_dim_expr_attrs;
  cinn::dialect::GenerateShapeOp::SymbolBindings lhs_symbol_bindings;
  bool success =
      cinn::dialect::MakeGenerateShapeOpAttribute(builder.ir_context(),
                                                  ShapeOrDataDimExprs4Value,
                                                  {lhs_expr},
                                                  group_inputs,
                                                  &lhs_minimal_inputs,
                                                  &lhs_output_dim_expr_attrs,
                                                  &lhs_symbol_bindings);
  CHECK(success);
  std::vector<pir::Value> rhs_minimal_inputs;
  std::vector<pir::Attribute> rhs_output_dim_expr_attrs;
  cinn::dialect::GenerateShapeOp::SymbolBindings rhs_symbol_bindings;
  success =
      cinn::dialect::MakeGenerateShapeOpAttribute(builder.ir_context(),
                                                  ShapeOrDataDimExprs4Value,
                                                  {rhs_expr},
                                                  group_inputs,
                                                  &rhs_minimal_inputs,
                                                  &rhs_output_dim_expr_attrs,
                                                  &rhs_symbol_bindings);
  CHECK(success);

  auto out_type = paddle::dialect::DenseTensorType::get(
      builder.ir_context(),
      pir::Int64Type::get(builder.ir_context()),
      ::common::make_ddim({1}));

  auto lhs_value =
      builder
          .Build<cinn::dialect::GenerateShapeOp>(lhs_minimal_inputs,
                                                 lhs_output_dim_expr_attrs,
                                                 lhs_symbol_bindings,
                                                 out_type)
          .out();
  auto rhs_value =
      builder
          .Build<cinn::dialect::GenerateShapeOp>(rhs_minimal_inputs,
                                                 rhs_output_dim_expr_attrs,
                                                 rhs_symbol_bindings,
                                                 out_type)
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

OpLoweringGroupPtr CloneGroup(const OpLoweringGroupPtr& group,
                              pir::Block* block,
                              pir::IrMapping* ir_mapping) {
  return group->Clone(block, ir_mapping);
}

void SetLeafBlockByGroupView(
    const OpLoweringGroupPtr& origin_group,
    const cinn::common::BroadcastLeaf& value_dim_exprs_list,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    pir::Builder& builder,  // NOLINT
    pir::Block* block,
    std::unordered_map<pir::Block*, OpLoweringGroupPtr>* group_map) {
  pir::IrMapping ir_mapping;
  auto origin_group_inputs = GetBlockOutsideInput(origin_group->ops());
  for (auto input : origin_group_inputs) {
    ir_mapping.Add(input, input);
  }

  auto new_group = CloneGroup(origin_group, block, &ir_mapping);
  new_group->SetIsBroadcastLeaf(true);
  PADDLE_ENFORCE_EQ(
      origin_group->ops().size(),
      new_group->ops().size(),
      phi::errors::InvalidArgument(
          "The size of origin group ops and new group ops is not equal,"
          "where the size of origin group ops:%d but the size of new group "
          "ops:%d.",
          origin_group->ops().size(),
          new_group->ops().size()));
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
    const OpLoweringGroupPtr& origin_group,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    const std::vector<pir::Value>& group_inputs,
    const std::vector<pir::Type>& output_types,
    pir::Builder& builder,  // NOLINT
    pir::Block* block,
    std::unordered_map<pir::Block*, OpLoweringGroupPtr>* group_map) {
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
            branch.Get<0>(), ShapeOrDataDimExprs4Value, group_inputs, builder);

    // lhs == rhs
    auto lhs_eq_rhs_cond_op = builder.Build<paddle::dialect::IfOp>(
        lhs_eq_rhs_cond, std::vector<pir::Type>{output_types});
    pir::Block& lhs_eq_rhs_block = lhs_eq_rhs_cond_op.true_block();
    builder.SetInsertionPointToBlockEnd(&lhs_eq_rhs_block);
    auto* lhs_eq_rhs_block_op = CreateConditionBlock(branch.Get<1>(),
                                                     origin_group,
                                                     ShapeOrDataDimExprs4Value,
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
                                                     ShapeOrDataDimExprs4Value,
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
                                                     ShapeOrDataDimExprs4Value,
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

void SimplyConditionBlock(
    pir::PatternRewriter& rewriter,  // NOLINT
    std::unordered_map<pir::Block*, OpLoweringGroupPtr>* group_map) {
  VLOG(4) << "simply condition block";
  using DoEachMutBlockGroupT =
      std::function<void(pir::Block*, const OpLoweringGroupPtr&)>;
  const auto& ForEachMutBlockGroup = [&](const DoEachMutBlockGroupT& DoEach) {
    for (auto& [block, group] : *group_map) {
      DoEach(block, group);
      std::vector<pir::Operation*> group_new_ops;
      group_new_ops.reserve(block->size());
      for (auto& op : *block) {
        if (!op.isa<pir::YieldOp>()) {
          group_new_ops.push_back(&op);
        }
      }
      group->SetOps(group_new_ops);
    }
  };
  ForEachMutBlockGroup([&](auto* block, const auto& group) {
    auto GetShapeOrDataForValue =
        [&group](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
      return group->GetShapeOrDataExprs(value);
    };
    EraseUnnecessaryExpandsInBlock(block, rewriter, GetShapeOrDataForValue);
  });
}
}  // namespace

namespace cinn::dialect::ir::details {

std::shared_ptr<BroadcastTree> ConstructBroadcastTree(
    const cinn::common::BroadcastLeaf& leaves) {
  VLOG(6) << "before constructed. broadcast-leaf: \n"
          << ToTxtString(cinn::common::BroadcastTree(leaves));
  int num_of_leaves = 0;
  auto broadcast_tree = std::make_shared<cinn::common::BroadcastTree>(
      cinn::common::ConstructBroadcastTree(cinn::common::BroadcastLeaf(leaves),
                                           &num_of_leaves));
  VLOG(4) << "num of broadcast tree leaves:" << num_of_leaves;
  VLOG(4) << "broadcast-tree: \n" << ToTxtString(*broadcast_tree);
  return broadcast_tree;
}

GroupDimExprInfo GetGroupDimExprInfo(const OpLoweringGroupPtr& group) {
  std::unordered_set<pir::Value> value_view;
  group->WalkOps([&group, &value_view](pir::Operation* op) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      value_view.insert(op->operand_source(i));
    }
    for (size_t i = 0; i < op->num_results(); ++i) {
      value_view.insert(op->result(i));
    }
  });

  GroupDimExprInfo group_dim_expr_info;
  for (auto value : value_view) {
    const auto& shape_dim_expr = group->GetShapeOrDataExprs(value);
    const auto& data_shape = shape_dim_expr.data();
    if (data_shape) {
      group_dim_expr_info.all_value_dim_exprs->push_back(*data_shape);
    } else {
      group_dim_expr_info.all_value_dim_exprs->push_back(
          shape_dim_expr.shape());
    }
    group_dim_expr_info.value_to_dim_expr_idx[value] =
        group_dim_expr_info.all_value_dim_exprs->size() - 1;
  }
  return group_dim_expr_info;
}

bool NeedBroadcastWithCF(const OpLoweringGroupPtr& group) {
  GroupDimExprInfo group_dim_expr_info = GetGroupDimExprInfo(group);
  const auto& leaves = group_dim_expr_info.all_value_dim_exprs;
  return NeedBroadcastWithCF(leaves);
}

bool NeedBroadcastWithCF(const cinn::common::BroadcastLeaf& leaves) {
  std::optional<symbol::Broadcastable<symbol::DimExpr>>
      broadcastable_condition = cinn::common::GetFirstCstrBroadcastable(leaves);
  return broadcastable_condition.has_value();
}

pir::Operation* CompileBroadcastTreeToConditionBlock(
    const OpLoweringGroupPtr& group,
    const BroadcastTree& broadcast_tree,
    const std::unordered_map<pir::Value, size_t>& value_to_dim_expr_idx,
    const std::vector<pir::Value>& group_inputs,
    const std::vector<pir::Type>& output_types,
    pir::PatternRewriter& rewriter) {  // NOLINT
  auto ShapeOrDataDimExprs4Value =
      [&group](pir::Value value) -> const symbol::ShapeOrDataDimExprs& {
    return group->GetShapeOrDataExprs(value);
  };
  // 1. broadcast tree to condition op
  VLOG(4) << "broadcast tree to condition op";
  std::unordered_map<pir::Block*, OpLoweringGroupPtr> group_map;
  pir::Operation* cond_op = CreateConditionBlock(broadcast_tree,
                                                 group,
                                                 ShapeOrDataDimExprs4Value,
                                                 value_to_dim_expr_idx,
                                                 group_inputs,
                                                 output_types,
                                                 rewriter,
                                                 rewriter.block(),
                                                 &group_map);

  // 2. compile condition block to jit_kernel_op
  CompileGroupToJitKernelOp(rewriter, &group_map);

  auto* program = group->ops().front()->GetParentProgram();
  VLOG(6) << "compile condition block to jit_kernel_op: " << *program;

  return cond_op;
}
}  // namespace cinn::dialect::ir::details
