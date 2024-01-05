#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/value.h"

namespace cinn::dialect {

namespace {

bool RunningFirst(cinn::dialect::GenerateShapeOp op, pir::Block* block) {
  for (int i = 0; i < op->num_operands(); ++i) {
    pir::Value input = op->operand_source(i);
    if (std::find(block->args_begin(), block->args_end(), input) == block->args_end()) return false;
  }
  return true;
}

const std::vector<symbol::DimExpr>& GetDimExprs(pir::Value value, const ShapeOrDataDimExprsAccessor& dim_exprs_accessor) {
  const auto& shape_or_data_dim_exprs = dim_exprs_accessor.GetShapeOrDataDimExprs(value);
  CHECK(shape_or_data_dim_exprs.data().has_value());
  return shape_or_data_dim_exprs.data().value();
}

// Returns `out` of GenerateShapeOp
pir::Value InsertGenerateShapeOpToRunFirst(
    pir::IrContext* ir_context,
    pir::Block* block,
    pir::Value value,
    const ShapeOrDataDimExprsAccessor& dim_exprs_accessor) {
  const auto& out_dim_exprs = GetDimExprs(value, dim_exprs_accessor);
  std::vector<pir::Value> minial_inputs{};
  std::vector<pir::Attribute> output_dim_expr_attrs{};
  cinn::dialect::GenerateShapeOp::SymbolBindings symbol_bindings{};
  MakeGenerateShapeOpAttribute(
    ir_context,
    dim_exprs_accessor.GetShapeOrDataDimExprs,
    out_dim_exprs,
    block->args(),
    &minial_inputs,
    &output_dim_expr_attrs,
    &symbol_bindings);
  pir::Builder builder(ir_context, block);
  return builder.Build<cinn::dialect::GenerateShapeOp>(minial_inputs, output_dim_expr_attrs, symbol_bindings).out();
}

void CloneDimExprInfo(pir::Value from, pir::Value to, const ShapeOrDataDimExprsAccessor& ctx) {
  ctx.SetShapeOrDataDimExprs(to, ctx.GetShapeOrDataDimExprs(from));
}

void ReplaceAllUses(pir::Value from, pir::Value to) {
  from.ReplaceAllUsesWith(to);
}

void EraseGenerateShapeOp(pir::Block::ConstIterator op_iter, pir::Block* block) {
  block->erase(op_iter);
}

bool RewriteOneGenerateShapeOpToRunFirst(
    pir::IrContext* ir_context,
    pir::Block* block,
    const ShapeOrDataDimExprsAccessor& dim_exprs_accessor) {
  for (auto op_iter = block->begin(); op_iter != block->end(); ++op_iter) {
    if (!op_iter->isa<cinn::dialect::GenerateShapeOp>()) continue;
    auto op = op_iter->dyn_cast<cinn::dialect::GenerateShapeOp>();
    if (RunningFirst(op, block)) continue;
    pir::Value new_shape = InsertGenerateShapeOpToRunFirst(ir_context, block, op.out(), dim_exprs_accessor);
    CloneDimExprInfo(op.out(), new_shape, dim_exprs_accessor);
    ReplaceAllUses(op.out(), new_shape);
    EraseGenerateShapeOp(op_iter, block);
    return true;
  }
  return false;
}

}

bool RewriteGenerateShapeOpToRunFirst(
    pir::IrContext* ir_context,
    pir::Block* block,
    const ShapeOrDataDimExprsAccessor& dim_exprs_accessor) {
  bool rewrited = false;
  while (RewriteOneGenerateShapeOpToRunFirst(ir_context, block, dim_exprs_accessor)) {
    rewrited = true;
  }
  return rewrited;
}

}