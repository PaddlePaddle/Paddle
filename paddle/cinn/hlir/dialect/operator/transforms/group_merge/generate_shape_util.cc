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
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/value.h"

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
  CHECK(shape_or_data_dim_exprs.data().has_value());
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
pir::Value InsertGenerateShapeOpToRunFirst(
    pir::Builder* builder,
    const std::vector<pir::Value>& block_args,
    pir::Value value,
    const ShapeOrDataDimExprsAccessor& dim_exprs_accessor) {
  const auto& out_dim_exprs = GetDimExprs(value, dim_exprs_accessor);
  std::vector<pir::Value> minial_inputs{};
  std::vector<pir::Attribute> output_dim_expr_attrs{};
  cinn::dialect::GenerateShapeOp::SymbolBindings symbol_bindings{};
  MakeGenerateShapeOpAttribute(builder->ir_context(),
                               dim_exprs_accessor.GetShapeOrDataDimExprs,
                               out_dim_exprs,
                               block_args,
                               &minial_inputs,
                               &output_dim_expr_attrs,
                               &symbol_bindings);
  return builder
      ->Build<cinn::dialect::GenerateShapeOp>(
          minial_inputs, output_dim_expr_attrs, symbol_bindings)
      .out();
}

void CloneDimExprInfo(pir::Value from,
                      pir::Value to,
                      const ShapeOrDataDimExprsAccessor& ctx) {
  ctx.SetShapeOrDataDimExprs(to, ctx.GetShapeOrDataDimExprs(from));
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
    pir::Value new_shape = InsertGenerateShapeOpToRunFirst(
        &builder, block_args, op.out(), dim_exprs_accessor);
    CloneDimExprInfo(op.out(), new_shape, dim_exprs_accessor);
    ReplaceAllUses(op.out(), new_shape);
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
  bool rewrited = false;
  while (RewriteOneGenerateShapeOpToRunFirst(
      ir_context, block, dim_exprs_accessor)) {
    rewrited = true;
  }
  return rewrited;
}

}  // namespace cinn::dialect
