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

#include "paddle/cinn/hlir/dialect/operator/transforms/accuracy_check_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_to_pd_util.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"
namespace cinn::dialect::ir {

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

using SymbolName2CachedValue = std::unordered_map<symbol::DimExpr, pir::Value>;

struct CachedDimExprToValueConverter {
  CachedDimExprToValueConverter(
      const TensorDim4SymbolNameT& TensorDim4SymbolNameVal,
      pir::PatternRewriter* rewriter_val)
      : TensorDim4SymbolName(TensorDim4SymbolNameVal), rewriter(rewriter_val) {}

  TensorDim4SymbolNameT TensorDim4SymbolName;
  pir::PatternRewriter* rewriter;

  pir::Value ConvertToValue(const symbol::DimExpr& dim_expr) {
    pir::Value value =
        std::visit([&](const auto& impl) { return ConvertToValueImpl(impl); },
                   dim_expr.variant());
    return value;
  }

  pir::Value GetInputShapeByInputTensor(pir::Value input_tensor) {
    auto iter = tensor2shape_.find(input_tensor);
    if (iter == tensor2shape_.end()) {
      pir::Value shape =
          rewriter->Build<paddle::dialect::ShapeOp>(input_tensor).out();
      pir::Value cast_shape =
          rewriter->Build<paddle::dialect::CastOp>(shape, phi::DataType::INT64)
              .out();
      iter = tensor2shape_.emplace(input_tensor, cast_shape).first;
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
    auto CastToInt64IfNeed = [&](pir::Value value) {
      if (value.type()
              .dyn_cast<paddle::dialect::DenseTensorType>()
              .dtype()
              .isa<pir::Int64Type>()) {
        return value;
      }
      return rewriter
          ->Build<paddle::dialect::CastOp>(value, phi::DataType::INT64)
          .out();
    };
    auto FlattenValueIfNeed = [&](pir::Value value) {
      const auto& dims =
          value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
      if (dims.size() <= 1) {
        return value;
      }
      return rewriter
          ->Build<paddle::dialect::FlattenOp>(value, 0, dims.size() - 1)
          .out();
    };
    if (tensor_dim.value.type()
            .dyn_cast<paddle::dialect::DenseTensorType>()
            .dims()
            .size() == 0) {
      return CastToInt64IfNeed(
          rewriter
              ->Build<paddle::dialect::ReshapeOp>(tensor_dim.value,
                                                  std::vector<int64_t>{1})
              .out());
    }
    return CastToInt64IfNeed(rewriter
                                 ->Build<paddle::dialect::SliceOp>(
                                     FlattenValueIfNeed(tensor_dim.value),
                                     std::vector<int64_t>{0LL},
                                     std::vector<int64_t>{tensor_dim.axis},
                                     std::vector<int64_t>{tensor_dim.axis + 1},
                                     std::vector<int64_t>{},
                                     std::vector<int64_t>{})
                                 .out());
  }

  pir::Value ConvertToValueImpl(
      const symbol::Negative<symbol::DimExpr>& dim_expr) {
    PADDLE_THROW(
        phi::errors::Fatal("Dead code. This logical should handled by "
                           "ConvertToValueImpl(symbol::Add<symbol::DimExpr>)"));
  }

  pir::Value ConvertToValueImpl(
      const symbol::Reciprocal<symbol::DimExpr>& dim_expr) {
    PADDLE_THROW(
        phi::errors::Fatal("Dead code. This logical should handled by "
                           "ConvertToValueImpl(symbol::Mul<symbol::DimExpr>)"));
  }

  pir::Value ConvertToValueImpl(const symbol::Add<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      phi::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
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
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      phi::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value prod = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      if (operands->at(i).isa<symbol::Reciprocal<symbol::DimExpr>>()) {
        const auto& operand =
            operands->at(i)
                .dyn_cast<symbol::Reciprocal<symbol::DimExpr>>()
                ->data;
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
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      phi::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value max = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      max =
          rewriter->Build<paddle::dialect::MaximumOp>(max, operand_value).out();
    }
    return max;
  }

  pir::Value ConvertToValueImpl(const symbol::Min<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      phi::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
    pir::Value min = ConvertToValue(operands->at(0));
    for (int i = 1; i < operands->size(); ++i) {
      pir::Value operand_value = ConvertToValue(operands->at(i));
      min =
          rewriter->Build<paddle::dialect::MinimumOp>(min, operand_value).out();
    }
    return min;
  }

  pir::Value ConvertToValueImpl(
      const symbol::Broadcast<symbol::DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    PADDLE_ENFORCE_GT(operands->size(),
                      0,
                      phi::errors::InvalidArgument(
                          "The size of operands is incorrect."
                          "Expected size is larger than 0, but receive %d.",
                          operands->size()));
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

pir::Value GetValueOfRewrittenOps(const std::vector<symbol::DimExpr>& dim_exprs,
                                  CachedDimExprToValueConverter* converter) {
  const std::vector<pir::Value>& values_from_dim_exprs =
      GetValuesOfRewrittenOps(dim_exprs, converter);
  if (values_from_dim_exprs.size() == 1) return values_from_dim_exprs.at(0);
  pir::Value vec =
      converter->rewriter->Build<pir::CombineOp>(values_from_dim_exprs).out();
  return converter->rewriter->Build<paddle::dialect::ConcatOp>(vec).out();
}
std::optional<pir::Value> GetOutReplacement(cinn::dialect::GenerateShapeOp op,
                                            pir::PatternRewriter* rewriter) {
  std::vector<symbol::DimExpr> dim_exprs = GetOutDimExprs(op);
  TensorDim4SymbolNameT TensorDim4SymbolName =
      MakeGetterTensorDim4SymbolName(op);
  if (!TensorDim4SymbolName) return std::nullopt;
  CachedDimExprToValueConverter converter{TensorDim4SymbolName, rewriter};
  return GetValueOfRewrittenOps(dim_exprs, &converter);
}

class AddAccuracyCheckPattern
    : public pir::OpRewritePattern<cinn::dialect::FusionOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::FusionOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::FusionOp fusion_op,
                       pir::PatternRewriter& rewriter) const override {
    const auto op_list = fusion_op.GetOperators();

    const auto group_info = fusion_op.attribute("group_info")
                                .dyn_cast<cinn::dialect::GroupInfoAttribute>()
                                .data();
    const auto& fn_name = group_info.fn_name;

    ::pir::IrMapping ir_mapping;
    ::pir::CloneOptions clone_options(/*clone_regions=*/false,
                                      /*clone_operands=*/true,
                                      /*clone_successors=*/false);
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
    ::pir::Builder builder = ::pir::Builder(ctx, fusion_op->GetParent());

    const auto& InsertAccuaryCheckOp = [&](::pir::Operation* op) -> void {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        rewriter.Build<paddle::dialect::AccuracyCheckOp>(
            fusion_op.result(i),
            ir_mapping.Lookup(op->operand_source(i)),
            fn_name,
            i);
      }
    };

    const auto& ConvertCinnOpToPdOp = [&](::pir::Operation* op) -> void {
      if (op->isa<cinn::dialect::GenerateShapeOp>()) {
        auto cinn_op = op->dyn_cast<cinn::dialect::GenerateShapeOp>();
        std::optional<pir::Value> out_replacement =
            details::GetOutReplacement(cinn_op, &rewriter);
        if (!out_replacement.has_value()) return;
        ir_mapping.Add(op->result(0), out_replacement.value());
        rewriter.SetInsertionPointAfter(out_replacement.value().defining_op());
        builder.SetInsertionPointAfter(out_replacement.value().defining_op());
        return;
      }
      for (size_t i = 0; i < op->num_operands(); ++i) {
        if (!ir_mapping.GetMap<pir::Value>().count(op->operand_source(i))) {
          ir_mapping.Add(op->operand_source(i), op->operand_source(i));
        }
      }
      pir::Operation* pd_op =
          cinn::dialect::details::RewriteCinnOpToPdOp(op, ir_mapping, builder);
      rewriter.SetInsertionPointAfter(pd_op);
      builder.SetInsertionPointAfter(pd_op);
    };

    const auto& ClonePdOp = [&](::pir::Operation* op) -> void {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        if (!ir_mapping.GetMap<pir::Value>().count(op->operand_source(i))) {
          ir_mapping.Add(op->operand_source(i), op->operand_source(i));
        }
      }
      auto new_op = op->Clone(ir_mapping, clone_options);
      rewriter.Insert(new_op);
      rewriter.SetInsertionPointAfter(new_op);
      builder.SetInsertionPointAfter(new_op);
    };

    rewriter.SetInsertionPointAfter(fusion_op);
    builder.SetInsertionPointAfter(fusion_op);
    for (auto& op : op_list) {
      if (op->isa<::pir::YieldOp>()) {
      } else if (op->dialect()->name() == "cinn_op") {
        ClonePdOp(op);
      }
    }
    return true;
  }
};

class AccuarcyCheckPass : public pir::Pass {
 public:
  AccuarcyCheckPass() : pir::Pass("accuracy_check_pass", /*opt_level=*/1) {}

  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<AddAccuracyCheckPattern>(context);

    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    int64_t num_ops{0};
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      auto& region = op->region(i);
      for (auto& block : region) {
        num_ops += block.size();
      }
    }
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 1;
    auto [_, num_rewrites] = pir::ApplyPatternsGreedily(op, patterns_, cfg);
    AddStatistics(num_rewrites, num_ops);
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }

 private:
  pir::FrozenRewritePatternSet patterns_;
};

std::unique_ptr<pir::Pass> CreateAccuarcyCheckPass() {
  return std::make_unique<AccuarcyCheckPass>();
}

}  // namespace cinn::dialect::ir
