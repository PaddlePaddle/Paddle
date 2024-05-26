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

#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/common/enforce.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr_util.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

std::unique_ptr<pir::Pass> CreateSplitGenerateShapeIntoShapeOpsPass() {
  return std::make_unique<SplitGenerateShapeIntoShapeOpsPass>();
}

bool SplitGenerateShapeIntoShapeOps::MatchAndRewrite(
    cinn::dialect::GenerateShapeOp op, pir::PatternRewriter& rewriter) const {
  std::optional<pir::Value> out_replacement =
      details::GetOutReplacement(op, &rewriter);
  if (!out_replacement.has_value()) return false;
  rewriter.ReplaceAllUsesWith(op->result(0), out_replacement.value());
  if (op->use_empty()) {
    rewriter.EraseOp(op);
  }
  return true;
}

namespace details {

std::optional<pir::Value> GetOutReplacement(cinn::dialect::GenerateShapeOp op,
                                            pir::PatternRewriter* rewriter) {
  std::vector<symbol::DimExpr> dim_exprs = GetOutDimExprs(op);
  TensorDim4SymbolNameT TensorDim4SymbolName =
      MakeGetterTensorDim4SymbolName(op);
  if (!TensorDim4SymbolName) return std::nullopt;
  CachedDimExprToValueConverter converter{TensorDim4SymbolName, rewriter};
  return GetValueOfRewrittenOps(dim_exprs, &converter);
}

TensorDim4SymbolNameT MakeGetterTensorDim4SymbolName(
    cinn::dialect::GenerateShapeOp op) {
  std::unordered_map<std::string, TensorDim> symbol_name2tensor_dim{};
  const auto& attr_map = op->attributes();
  const auto& iter = attr_map.find("symbol_bindings");
  PADDLE_ENFORCE((iter != attr_map.end()),
                 phi::errors::PreconditionNotMet(
                     "attr symbol_bindings MUST in attribute map for [%s] op",
                     op->name()));
  pir::Attribute attr = iter->second;
  auto* Convert =
      &cinn::dialect::GenerateShapeOp::ConvertAttributeToSymbolBindings;
  const auto& symbol_bindings = Convert(attr);
  PADDLE_ENFORCE(
      symbol_bindings.has_value(),
      phi::errors::PreconditionNotMet("attr symbol_bindings in op [%s] can "
                                      "not be converted to symbol bindings",
                                      op->name()));
  for (const auto& symbol_binding : symbol_bindings.value()) {
    InsertSymbolBinding(op, symbol_binding, &symbol_name2tensor_dim);
  }
  return [map = std::move(symbol_name2tensor_dim)](
             const std::string& symbol_name) -> std::optional<TensorDim> {
    auto iter = map.find(symbol_name);
    if (iter == map.end()) return std::nullopt;
    return iter->second;
  };
}

void InsertSymbolBinding(
    cinn::dialect::GenerateShapeOp op,
    const cinn::dialect::GenerateShapeOp::SymbolBinding& symbol_binding,
    std::unordered_map<std::string, TensorDim>* symbol_name2tensor_dim) {
  return std::visit(
      [&](const auto& impl) {
        return InsertSymbolBindingImpl(op, impl, symbol_name2tensor_dim);
      },
      symbol_binding);
}

void InsertSymbolBindingImpl(
    cinn::dialect::GenerateShapeOp op,
    const cinn::dialect::GenerateShapeOp::DataSymbolBinding& symbol_binding,
    std::unordered_map<std::string, TensorDim>* symbol_name2tensor_dim) {
  (*symbol_name2tensor_dim)[symbol_binding.symbol_name] = TensorDimInData{
      .value = op.operand_source(symbol_binding.input_tensor_idx),
      .axis = symbol_binding.input_tensor_dim_idx};
}

void InsertSymbolBindingImpl(
    cinn::dialect::GenerateShapeOp op,
    const cinn::dialect::GenerateShapeOp::ShapeSymbolBinding& symbol_binding,
    std::unordered_map<std::string, TensorDim>* symbol_name2tensor_dim) {
  (*symbol_name2tensor_dim)[symbol_binding.symbol_name] = TensorDimInShape{
      .value = op.operand_source(symbol_binding.input_tensor_idx),
      .axis = symbol_binding.input_tensor_dim_idx};
}

std::vector<symbol::DimExpr> GetOutDimExprs(cinn::dialect::GenerateShapeOp op) {
  const auto& attr_map = op->attributes();
  const auto& iter = attr_map.find("output_dim_exprs");
  PADDLE_ENFORCE((iter != attr_map.end()),
                 phi::errors::PreconditionNotMet(
                     "attr output_dim_exprs MUST in attribute map for [%s] op",
                     op->name()));
  pir::Attribute output_dim_exprs_attr = iter->second;
  PADDLE_ENFORCE(
      output_dim_exprs_attr.isa<pir::ArrayAttribute>(),
      phi::errors::PreconditionNotMet(
          "attr output_dim_exprs for [%s] op must be an pir::ArrayAttribute",
          op->name()));
  std::vector<symbol::DimExpr> ret{};
  const auto& output_dim_exprs =
      output_dim_exprs_attr.dyn_cast<pir::ArrayAttribute>();
  for (int i = 0; i < output_dim_exprs.size(); ++i) {
    const auto& attr = output_dim_exprs.at(i);
    const auto& opt_dim_expr = cinn::dialect::ConvertAttributeToDimExpr(attr);
    CHECK(opt_dim_expr.has_value());
    ret.emplace_back(opt_dim_expr.value());
  }
  return ret;
}

pir::Value GetValueOfRewrittenOps(const std::vector<symbol::DimExpr>& dim_exprs,
                                  CachedDimExprToValueConverter* converter) {
  const std::vector<pir::Value>& values_from_dim_exprs =
      GetValuesOfRewrittenOps(dim_exprs, converter);
  if (values_from_dim_exprs.size() == 1) return values_from_dim_exprs.at(0);
  pir::Value vec =
      converter->rewriter->Build<pir::CombineOp>(values_from_dim_exprs).out();
  return converter->rewriter->Build<paddle::dialect::ConcatOp>(vec).out();
}

std::vector<pir::Value> GetValuesOfRewrittenOps(
    const std::vector<symbol::DimExpr>& dim_exprs,
    CachedDimExprToValueConverter* converter) {
  std::vector<pir::Value> ret;
  for (const auto& dim_expr : dim_exprs) {
    const auto& simplified = symbol::SimplifyDimExpr(dim_expr);
    pir::Value value = converter->ConvertToValue(simplified);
    ret.push_back(value);
  }
  return ret;
}
}  // namespace details

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
