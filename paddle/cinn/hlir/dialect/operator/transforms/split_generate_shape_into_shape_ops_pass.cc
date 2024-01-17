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

#include "paddle/cinn/common/dim_expr_simplify.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

struct TensorDimInShape {
  pir::Value value;
  int axis;
};

struct TensorDimInData {
  pir::Value value;
  int axis;
};

using TensorDim = std::variant<TensorDimInShape, TensorDimInData>;

using TensorDim4SymbolNameT =
    std::function<std::optional<TensorDim>(const std::string& symbol_name)>;

struct CachedDimExprToValueConverter {
  CachedDimExprToValueConverter(
      const TensorDim4SymbolNameT& TensorDim4SymbolNameVal,
      pir::PatternRewriter* rewriter_val)
      : TensorDim4SymbolName(TensorDim4SymbolNameVal), rewriter(rewriter_val) {}

  TensorDim4SymbolNameT TensorDim4SymbolName;
  pir::PatternRewriter* rewriter;

  // TODO(): Refactor to cached version if std::hash<symbol::DimExpr>() is
  // ready. std::unordered_map<symbol::DimExpr, pir::Value>
  // symbol_names2cached_value_;

  pir::Value ConvertToValue(const symbol::DimExpr& dim_expr) {
    // TODO():  cache the returned value if std::hash<symbol::DimExpr>() is
    // ready
    return std::visit(
        [&](const auto& impl) { return ConvertToValueImpl(impl); },
        dim_expr.variant());
  }

  pir::Value GetInputShapeByInputTensor(pir::Value input_tensor) {
    auto iter = tensor2shape_.find(input_tensor);
    if (iter == tensor2shape_.end()) {
      pir::Value input_shape =
          rewriter->Build<paddle::dialect::ShapeOp>(input_tensor).out();
      iter = tensor2shape_.emplace(input_tensor, input_shape).first;
    }
    return iter->second;
  }

 private:
  std::unordered_map<pir::Value /*input tensor*/,
                     pir::Value /*input shape tensor*/>
      tensor2shape_;

  pir::Value ConvertToValueImpl(int64_t dim_expr) {
    return rewriter
        ->Build<paddle::dialect::FullIntArrayOp>(std::vector{dim_expr},
                                                 phi::DataType::INT64)
        .out();
  }

  pir::Value ConvertToValueImpl(const std::string& symbol_name) {
    const auto& tensor_dim = TensorDim4SymbolName(symbol_name);
    PADDLE_ENFORCE(
        tensor_dim.has_value(),
        phi::errors::PreconditionNotMet(
            "symbol [%s] are not bound to any input of generate_shape op",
            symbol_name));
    return std::visit(
        [&](const auto& impl) { return ConvertTensorDimToValue(impl); },
        tensor_dim.value());
  }

  pir::Value ConvertTensorDimToValue(const TensorDimInShape& tensor_dim) {
    pir::Value input_shape = GetInputShapeByInputTensor(tensor_dim.value);
    return ConvertTensorDimToValue(
        TensorDimInData{.value = input_shape, .axis = tensor_dim.axis});
  }

  pir::Value ConvertTensorDimToValue(const TensorDimInData& tensor_dim) {
    return rewriter
        ->Build<paddle::dialect::SliceOp>(
            tensor_dim.value,
            std::vector<int64_t>{0LL},
            std::vector<int64_t>{tensor_dim.axis},
            std::vector<int64_t>{tensor_dim.axis + 1},
            std::vector<int64_t>{},
            std::vector<int64_t>{})
        .out();
  }

  pir::Value ConvertToValueImpl(
      const symbol::Negative<symbol::DimExpr>& dim_expr) {
    LOG(FATAL) << "Dead code. This logical should handled by "
                  "ConvertToValueImpl(symbol::Add<symbol::DimExpr>)";
  }

  pir::Value ConvertToValueImpl(
      const symbol::Reciprocal<symbol::DimExpr>& dim_expr) {
    LOG(FATAL) << "Dead code. This logical should handled by "
                  "ConvertToValueImpl(symbol::Mul<symbol::DimExpr>)";
  }

  pir::Value ConvertToValueImpl(const symbol::Add<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    CHECK_GT(operands->size(), 0);
    pir::Value acc = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      if (operands->at(i).isa<symbol::Negative<symbol::DimExpr>>()) {
        const auto& [operand] =
            *operands->at(i).dyn_cast<symbol::Negative<symbol::DimExpr>>();
        pir::Value operand_value = ConvertToValue(operand);
        acc = rewriter->Build<paddle::dialect::SubtractOp>(acc, operand_value)
                  .out();
      } else {
        pir::Value operand_value = ConvertToValue(operands->at(i));
        acc = rewriter->Build<paddle::dialect::AddOp>(acc, operand_value).out();
      }
    }
    return acc;
  }

  pir::Value ConvertToValueImpl(const symbol::Mul<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    CHECK_GT(operands->size(), 0);
    pir::Value prod = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      if (operands->at(i).isa<symbol::Reciprocal<symbol::DimExpr>>()) {
        const auto& [operand] =
            *operands->at(i).dyn_cast<symbol::Negative<symbol::DimExpr>>();
        pir::Value operand_value = ConvertToValue(operand);
        prod = rewriter->Build<paddle::dialect::DivideOp>(prod, operand_value)
                   .out();
      } else {
        pir::Value operand_value = ConvertToValue(operands->at(i));
        prod = rewriter->Build<paddle::dialect::MultiplyOp>(prod, operand_value)
                   .out();
      }
    }
    return prod;
  }

  pir::Value ConvertToValueImpl(const symbol::Max<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    CHECK_GT(operands->size(), 0);
    pir::Value max = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      max = rewriter->Build<paddle::dialect::MaxOp>(max, operand_value).out();
    }
    return max;
  }

  pir::Value ConvertToValueImpl(const symbol::Min<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    CHECK_GT(operands->size(), 0);
    pir::Value min = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      min = rewriter->Build<paddle::dialect::MinOp>(min, operand_value).out();
    }
    return min;
  }

  pir::Value ConvertToValueImpl(
      const symbol::Broadcast<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    CHECK_GT(operands->size(), 0);
    pir::Value broadcasted = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      broadcasted = rewriter
                        ->Build<paddle::dialect::ShapeBroadcastOp>(
                            broadcasted, operand_value)
                        .out();
    }
    return broadcasted;
  }
};

}  // namespace

class SplitGenerateShapeIntoShapeOps
    : public pir::OpRewritePattern<cinn::dialect::GenerateShapeOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::GenerateShapeOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::GenerateShapeOp op,
                       pir::PatternRewriter& rewriter) const override {
    std::optional<pir::Value> out_replacement =
        GetOutReplacement(op, &rewriter);
    if (!out_replacement.has_value()) return false;
    rewriter.ReplaceAllUsesWith(op->result(0), out_replacement.value());
    return true;
  }

  std::optional<pir::Value> GetOutReplacement(
      cinn::dialect::GenerateShapeOp op, pir::PatternRewriter* rewriter) const {
    std::vector<symbol::DimExpr> dim_exprs = GetOutDimExprs(op);
    TensorDim4SymbolNameT TensorDim4SymbolName =
        MakeGetterTensorDim4SymbolName(op);
    if (!TensorDim4SymbolName) return std::nullopt;
    CachedDimExprToValueConverter converter{TensorDim4SymbolName, rewriter};
    return GetValueOfRewritedOps(dim_exprs, &converter);
  }

  TensorDim4SymbolNameT MakeGetterTensorDim4SymbolName(
      cinn::dialect::GenerateShapeOp op) const {
    std::unordered_map<std::string, TensorDim> symbol_name2tenso_dim{};
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
      InsertSymbolBinding(op, symbol_binding, &symbol_name2tenso_dim);
    }
    return [map = std::move(symbol_name2tenso_dim)](
               const std::string& symbol_name) -> std::optional<TensorDim> {
      auto iter = map.find(symbol_name);
      if (iter == map.end()) return std::nullopt;
      return iter->second;
    };
  }

  void InsertSymbolBinding(
      cinn::dialect::GenerateShapeOp op,
      const cinn::dialect::GenerateShapeOp::SymbolBinding& symbol_binding,
      std::unordered_map<std::string, TensorDim>* symbol_name2tenso_dim) const {
    return std::visit(
        [&](const auto& impl) {
          return InsertSymbolBindingImpl(op, impl, symbol_name2tenso_dim);
        },
        symbol_binding);
  }

  void InsertSymbolBindingImpl(
      cinn::dialect::GenerateShapeOp op,
      const cinn::dialect::GenerateShapeOp::DataSymbolBinding& symbol_binding,
      std::unordered_map<std::string, TensorDim>* symbol_name2tenso_dim) const {
    (*symbol_name2tenso_dim)[symbol_binding.symbol_name] = TensorDimInData{
        .value = op.operand_source(symbol_binding.input_tensor_idx),
        .axis = symbol_binding.input_tensor_dim_idx};
  }

  void InsertSymbolBindingImpl(
      cinn::dialect::GenerateShapeOp op,
      const cinn::dialect::GenerateShapeOp::ShapeSymbolBinding& symbol_binding,
      std::unordered_map<std::string, TensorDim>* symbol_name2tenso_dim) const {
    (*symbol_name2tenso_dim)[symbol_binding.symbol_name] = TensorDimInShape{
        .value = op.operand_source(symbol_binding.input_tensor_idx),
        .axis = symbol_binding.input_tensor_dim_idx};
  }

  std::vector<symbol::DimExpr> GetOutDimExprs(
      cinn::dialect::GenerateShapeOp op) const {
    const auto& attr_map = op->attributes();
    const auto& iter = attr_map.find("output_dim_exprs");
    PADDLE_ENFORCE(
        (iter != attr_map.end()),
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

  pir::Value GetValueOfRewritedOps(
      const std::vector<symbol::DimExpr>& dim_exprs,
      CachedDimExprToValueConverter* converter) const {
    const std::vector<pir::Value>& values_from_dim_exprs =
        GetValuesOfRewritedOps(dim_exprs, converter);
    return converter->rewriter->Build<pir::CombineOp>(values_from_dim_exprs)
        .out();
  }

  std::vector<pir::Value> GetValuesOfRewritedOps(
      const std::vector<symbol::DimExpr>& dim_exprs,
      CachedDimExprToValueConverter* converter) const {
    std::vector<pir::Value> ret;
    for (const auto& dim_expr : dim_exprs) {
      const auto& simplified = cinn::common::SimplifyDimExpr(dim_expr);
      pir::Value value = converter->ConvertToValue(simplified);
      ret.push_back(value);
    }
    return ret;
  }
};

SplitGenerateShapeIntoShapeOpsPass::SplitGenerateShapeIntoShapeOpsPass()
    : pir::PatternRewritePass("split_generate_shape_into_shape_ops_pass", 1) {}

pir::RewritePatternSet SplitGenerateShapeIntoShapeOpsPass::InitializePatterns(
    pir::IrContext* context) {
  pir::RewritePatternSet ps(context);
  // elementwise ops
  ps.Add<SplitGenerateShapeIntoShapeOps>(context);
  return ps;
}

bool SplitGenerateShapeIntoShapeOpsPass::CanApplyOn(pir::Operation* op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
