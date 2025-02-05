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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/generate_shape_util.h"
#include <unordered_set>
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/value.h"

namespace cinn::dialect {

namespace {

bool RunningFirst(cinn::dialect::GenerateShapeOp op,
                  const std::vector<pir::Value>& block_args) {
  for (int i = 0; i < op->num_operands(); ++i) {
    pir::Value input = op->operand_source(i);
    if (std::find(block_args.begin(), block_args.end(), input) ==
        block_args.end())
      return false;
  }
  return true;
}

const std::vector<symbol::DimExpr>& GetDimExprs(
    pir::Value value, const ShapeOrDataDimExprsAccessor& dim_exprs_accessor) {
  const auto& shape_or_data_dim_exprs =
      dim_exprs_accessor.GetShapeOrDataDimExprs(value);
  PADDLE_ENFORCE_EQ(
      shape_or_data_dim_exprs.data().has_value(),
      true,
      ::common::errors::InvalidArgument(
          "shape_or_data_dim_exprs has no data, it cannot be empty"));
  return shape_or_data_dim_exprs.data().value();
}

std::vector<pir::Value> GetBlockArgs(pir::Block* block) {
  std::unordered_set<pir::Value> values_produced_by_block_op;
  for (auto op = block->begin(); op != block->end(); ++op) {
    for (int i = 0; i < op->num_results(); ++i) {
      values_produced_by_block_op.insert(op->result(i));
    }
  }
  std::vector<pir::Value> ret{};
  for (auto op = block->begin(); op != block->end(); ++op) {
    for (int i = 0; i < op->num_operands(); ++i) {
      pir::Value input = op->operand_source(i);
      if (!input.type().isa<pir::DenseTensorType>()) continue;
      if (values_produced_by_block_op.count(input) == 0) {
        if (std::find(ret.begin(), ret.end(), input) == ret.end()) {
          ret.push_back(input);
        }
      }
    }
  }
  return ret;
}

// Returns `out` of GenerateShapeOp
std::optional<pir::Value> InsertGenerateShapeOpToRunFirst(
    pir::Builder* builder,
    const std::vector<pir::Value>& block_args,
    pir::Value value,
    const ShapeOrDataDimExprsAccessor& dim_exprs_accessor) {
  const auto& out_dim_exprs = GetDimExprs(value, dim_exprs_accessor);
  std::vector<pir::Value> minimal_inputs{};
  std::vector<pir::Attribute> output_dim_expr_attrs{};
  cinn::dialect::GenerateShapeOp::SymbolBindings symbol_bindings{};
  bool success =
      MakeGenerateShapeOpAttribute(builder->ir_context(),
                                   dim_exprs_accessor.GetShapeOrDataDimExprs,
                                   out_dim_exprs,
                                   block_args,
                                   &minimal_inputs,
                                   &output_dim_expr_attrs,
                                   &symbol_bindings);
  if (success) {
    return builder
        ->Build<cinn::dialect::GenerateShapeOp>(minimal_inputs,
                                                output_dim_expr_attrs,
                                                symbol_bindings,
                                                value.type())
        .out();
  }
  return std::nullopt;
}

void ReplaceAllUses(pir::Value from, pir::Value to) {
  from.ReplaceAllUsesWith(to);
}

void EraseGenerateShapeOp(pir::Block::ConstIterator op_iter,
                          pir::Block* block) {
  block->erase(op_iter);
}

bool RewriteOneGenerateShapeOpToRunFirst(
    pir::IrContext* ir_context,
    pir::Block* block,
    const ShapeOrDataDimExprsAccessor& dim_exprs_accessor) {
  std::vector<pir::Value> block_args = GetBlockArgs(block);
  for (auto op_iter = block->begin(); op_iter != block->end(); ++op_iter) {
    if (!op_iter->isa<cinn::dialect::GenerateShapeOp>()) continue;
    auto op = op_iter->dyn_cast<cinn::dialect::GenerateShapeOp>();
    if (RunningFirst(op, block_args)) continue;
    pir::Builder builder(ir_context, block);
    builder.set_insertion_point(op);
    std::optional<pir::Value> new_shape = InsertGenerateShapeOpToRunFirst(
        &builder, block_args, op.out(), dim_exprs_accessor);
    if (!new_shape.has_value()) continue;
    ReplaceAllUses(op.out(), new_shape.value());
    EraseGenerateShapeOp(op_iter, block);
    return true;
  }
  return false;
}

}  // namespace

bool MoveGenerateShapeOpsToPrologue(
    pir::IrContext* ir_context,
    pir::Block* block,
    const ShapeOrDataDimExprsAccessor& dim_exprs_accessor) {
  bool rewritten = false;
  while (RewriteOneGenerateShapeOpToRunFirst(
      ir_context, block, dim_exprs_accessor)) {
    rewritten = true;
  }
  return rewritten;
}

}  // namespace cinn::dialect
