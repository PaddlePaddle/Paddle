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

#include "paddle/cinn/hlir/dialect/operator/transforms/fuse_shape_ops_into_generate_shape_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/api/match_context.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/common/bfs_walker.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include <algorithm>
#include <glog/logging.h>


namespace cinn {
namespace dialect {
namespace ir {

namespace {

using ShapeOrDataDimExprs4ValueT = std::function<const symbol::ShapeOrDataDimExprs&(pir::Value)>;

std::vector<pir::Value> FindSourceDenseTensorOfDimTensor(pir::Value shape, const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  std::vector<pir::Value> ret{};
  const auto& Emplace = [&](pir::Value value) {
    if (std::find(ret.begin(), ret.end(), value) != ret.end()) return;
    ret.emplace_back(ret);
  };
  const auto& ForEachInputValue = [&](pir::Value value, const std::function<void(pir::Value)>& Visit) {
    // find input dimension tensor;
    pir::Operation* owner = value.defining_op();
    if (owner == nullptr) return;
    for (int i = 0; i < owner->num_operands(); ++i) {
      Visit(owner->operand_source(i));
    }
  };
  const auto& IsDimTensor = [&](pir::Value value) -> bool {
    return ShapeOrDataDimExprs4Value(value).data().has_value();
  };
  const auto& ForEachInputDimTensor = [&](pir::Value value, const std::function<void(pir::Value)>& Visit) {
    // find input dimension tensor;
    ForEachInputValue(value, [&](pir::Value input) {
      if (IsDimTensor(input)) {
        Visit(input);
      }
    });
  };
  common::BfsWalker<pir::Value> walker(ForEachInputDimTensor);
  walker(shape, [&](pir::Value value) {
    size_t input_cnt = 0;
    ForEachInputValue(value, [&](pir::Value input) {
      ++input_cnt;
      if (IsDimTensor(input)) return;
      Emplace(input);
    });
    if (input_cnt == 0) {
      // `value` is a result of a source op.
      Emplace(value);
    }
  });
  return ret;
}

bool IsConstant(const std::vector<symbol::DimExpr>& dim_exprs) {
  for (const auto& dim_expr : dim_exprs) {
    if (dim_expr.isa<int64_t>()) continue;
    return false;
  }
  return true;
}

bool IsAtomicImpl(int64_t) {
  return true;
}

bool IsAtomicImpl(const std::string&) {
  return true;
}

bool IsAtomicImpl(const symbol::Negative<symbol::DimExpr>&) {
  return false;
}

bool IsAtomicImpl(const symbol::Reciprocal<symbol::DimExpr>&) {
  return false;
}

bool IsAtomicImpl(const symbol::Add<symbol::DimExpr>&) {
  return false;
}

bool IsAtomicImpl(const symbol::Mul<symbol::DimExpr>&) {
  return false;
}

bool IsAtomicImpl(const symbol::Max<symbol::DimExpr>&) {
  return false;
}

bool IsAtomicImpl(const symbol::Min<symbol::DimExpr>&) {
  return false;
}

bool IsAtomicImpl(const symbol::Broadcast<symbol::DimExpr>&) {
  return false;
}

bool IsAtomic(const symbol::DimExpr& dim_expr) {
  return std::visit([](const auto& impl) {
    return IsAtomicImpl(impl);
  }, dim_expr.variant());
}

bool InputDimExprsAllSupported(const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
                              const std::vector<pir::Value>& input_tensors) {
  const auto& AllSupported = [](const std::vector<symbol::DimExpr>& dim_exprs) -> bool {
    for (const auto& dim_expr : dim_exprs) {
      if (!IsAtomic(dim_expr)) return false;
    }
    return true;
  };
  for (const auto& input_tensor : input_tensors) {
    const auto& dim_exprs = ShapeOrDataDimExprs4Value(input_tensor);
    if (!AllSupported(dim_exprs.shape())) return false;
    if (dim_exprs.data().has_value()) {
      if (!AllSupported(dim_exprs.data().value())) return false;
    }
  }
  return true;
}

void ConvertDimExprToAttributes(pir::IrContext* ir_context,
                                const std::vector<symbol::DimExpr>& dim_exprs,
                                std::vector<pir::Attribute>* attrs) {
  attrs->clear();
  attrs->reserve(dim_exprs.size());
  for (const auto& dim_expr : dim_exprs) {
    attrs->emplace_back(ConvertDimExprToAttribute(ir_context, dim_expr));
  }
}

void CollectSymbolNames(const symbol::DimExpr& dim_expr, std::set<std::string>* ret);

void CollectSymbolNamesImpl(const int64_t& dim_expr, std::set<std::string>* ret) {
  // do nothing.
}

void CollectSymbolNamesImpl(const std::string& dim_expr, std::set<std::string>* ret) {
  ret->insert(dim_expr);
}

template <typename T>
void CollectSymbolNamesImplForUnary(const T& dim_expr, std::set<std::string>* ret) {
  const auto& [operand] = *dim_expr;
  CollectSymbolNames(operand, ret);
}

void CollectSymbolNamesImpl(const symbol::Negative<symbol::DimExpr>& dim_expr, std::set<std::string>* ret) {
  CollectSymbolNamesImplForUnary(dim_expr, ret);
}

void CollectSymbolNamesImpl(const symbol::Reciprocal<symbol::DimExpr>& dim_expr, std::set<std::string>* ret) {
  CollectSymbolNamesImplForUnary(dim_expr, ret);
}

template <typename T>
void CollectSymbolNamesImplForVariadic(const T& dim_expr, std::set<std::string>* ret) {
  const auto& [operands] = *dim_expr;
  for (const auto& operand : operands) {
    CollectSymbolNames(operand, ret);
  }
}

void CollectSymbolNamesImpl(const symbol::Add<symbol::DimExpr>& dim_expr, std::set<std::string>* ret) {
  CollectSymbolNamesImplForVariadic(dim_expr, ret);
}

void CollectSymbolNamesImpl(const symbol::Mul<symbol::DimExpr>& dim_expr, std::set<std::string>* ret) {
  CollectSymbolNamesImplForVariadic(dim_expr, ret);
}

void CollectSymbolNamesImpl(const symbol::Max<symbol::DimExpr>& dim_expr, std::set<std::string>* ret) {
  CollectSymbolNamesImplForVariadic(dim_expr, ret);
}

void CollectSymbolNamesImpl(const symbol::Min<symbol::DimExpr>& dim_expr, std::set<std::string>* ret) {
  CollectSymbolNamesImplForVariadic(dim_expr, ret);
}

void CollectSymbolNamesImpl(const symbol::Broadcast<symbol::DimExpr>& dim_expr, std::set<std::string>* ret) {
  CollectSymbolNamesImplForVariadic(dim_expr, ret);
}

void CollectSymbolNames(const symbol::DimExpr& dim_expr, std::set<std::string>* ret) {
  return std::visit([&](const auto& impl) {
    return CollectSymbolNamesImpl(impl, ret);
  }, dim_expr.variant());
}

void CollectSymbolNames(const std::vector<symbol::DimExpr>& dim_exprs, std::set<std::string>* ret) {
  for (const auto& dim_expr : dim_exprs) {
    CollectSymbolNames(dim_expr, ret);
  }
}

template<typename SymbolBindingsT>
void AppendSymbolBindings(const std::vector<symbol::DimExpr>& dim_exprs,
                          const std::set<std::string>& symbol_names,
                          int in_tensor_idx,
                          GenerateShapeOp::SymbolBindings* symbol_bindings) {
  for (int in_tensor_dim_idx = 0; in_tensor_dim_idx < dim_exprs.size(); ++in_tensor_dim_idx) {
    const auto& dim_expr = dim_exprs.at(in_tensor_dim_idx);
    CHECK(IsAtomic(dim_expr));
    if (!dim_expr.isa<std::string>()) continue;
    const auto& symbol_name = dim_expr.dyn_cast<std::string>();
    if (symbol_names.find(symbol_name) == symbol_names.end()) continue;
    symbol_bindings->emplace_back(SymbolBindingsT{
      .symbol_name=symbol_name,
      .input_tensor_idx=in_tensor_idx,
      .input_tensor_dim_idx=in_tensor_dim_idx,
    });
  }
}

void GenerateSymbolBindings(const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
                            const std::vector<pir::Value>& input_tensors,
                            const std::set<std::string>& symbol_names,
                            GenerateShapeOp::SymbolBindings* symbol_bindings) {
  for (int i = 0; i < input_tensors.size(); ++i) {
    const auto& input_tensor = input_tensors.at(i);
    const auto& dim_exprs = ShapeOrDataDimExprs4Value(input_tensor);
    AppendSymbolBindings<GenerateShapeOp::ShapeSymbolBinding>(
      dim_exprs.shape(), symbol_names, i, symbol_bindings);
    if (dim_exprs.data().has_value()) {
        AppendSymbolBindings<GenerateShapeOp::DataSymbolBinding>(
          dim_exprs.shape(), symbol_names, i, symbol_bindings);
    }
  }
}

bool MakeGenerateShapeOpAttribute(pir::IrContext* ir_context,
                                  const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value,
                                  const std::vector<pir::Value>& input_tensors,
                                  pir::Value output_shape,
                                  std::vector<pir::Attribute>* output_dim_expr_attrs,
                                  GenerateShapeOp::SymbolBindings* symbol_bindings) {
  const auto& shape_or_data_dim_exprs = ShapeOrDataDimExprs4Value(output_shape);
  CHECK(shape_or_data_dim_exprs.data().has_value());
  const auto& out_dim_exprs = shape_or_data_dim_exprs.data().value();
  if (IsConstant(out_dim_exprs)) return false;
  if (!InputDimExprsAllSupported(ShapeOrDataDimExprs4Value, input_tensors)) {
    VLOG(4) << "input dim_exprs are not as simple as symbols, please make sure they are handled by other passes";
    return false;
  }
  // generate output_dim_expr_attrs
  ConvertDimExprToAttributes(ir_context, out_dim_exprs, /*out*/output_dim_expr_attrs);
  // generate symbol_bindings
  std::set<std::string> symbol_names_in_out_dim_exprs{};
  CollectSymbolNames(out_dim_exprs, &symbol_names_in_out_dim_exprs);
  GenerateSymbolBindings(ShapeOrDataDimExprs4Value,
                         input_tensors,
                         symbol_names_in_out_dim_exprs,
                         /*out*/symbol_bindings);
  return true;
}

std::optional<pir::Value> GetOutOfRewritedGenerateShapeOp(
    pir::Value shape,
    pir::PatternRewriter* rewriter,
    const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  std::vector<pir::Value> input_tensors = FindSourceDenseTensorOfDimTensor(shape, ShapeOrDataDimExprs4Value);
  if (input_tensors.empty()) return std::nullopt;
  std::vector<pir::Attribute> output_dim_expr_attrs{};
  GenerateShapeOp::SymbolBindings symbol_bindings{};
  bool success = MakeGenerateShapeOpAttribute(rewriter->ir_context(),
                                              ShapeOrDataDimExprs4Value,
                                              input_tensors,
                                              shape,
                                              &output_dim_expr_attrs,
                                              &symbol_bindings);
  if (!success) return std::nullopt;
  return rewriter->Build<cinn::dialect::GenerateShapeOp>(input_tensors, output_dim_expr_attrs, symbol_bindings).out();
}

bool ProcessOp(paddle::dialect::ExpandOp op, pir::PatternRewriter* rewriter,
               const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value) {
  std::optional<pir::Value> opt_generated_shape =
    GetOutOfRewritedGenerateShapeOp(op.shape(), rewriter, ShapeOrDataDimExprs4Value);
  if (!opt_generated_shape.has_value()) return false;
  op->operand(1).set_source(opt_generated_shape.value());
  return true;
}

}

template <typename OPTYPE>
class FuseShapeOpsIntoGenerateShapeOpPattern : public pir::OpRewritePattern<OPTYPE> {
 public:

  FuseShapeOpsIntoGenerateShapeOpPattern(pir::IrContext *context, const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value)
    : pir::OpRewritePattern<OPTYPE>(context), ShapeOrDataDimExprs4Value_(ShapeOrDataDimExprs4Value) {}

  bool MatchAndRewrite(OPTYPE op,
                       pir::PatternRewriter& rewriter) const override {
    return ProcessOp(op, &rewriter, ShapeOrDataDimExprs4Value_);
  }

 private:
  ShapeOrDataDimExprs4ValueT ShapeOrDataDimExprs4Value_;
};

FuseShapeOpsIntoGenerateShapeOpPass::FuseShapeOpsIntoGenerateShapeOpPass(
  const ShapeOrDataDimExprs4ValueT& ShapeOrDataDimExprs4Value)
    : pir::PatternRewritePass("fuse_shape_ops_into_generate_shape_op_pass", 1),
      ShapeOrDataDimExprs4Value_(ShapeOrDataDimExprs4Value) {}

pir::RewritePatternSet FuseShapeOpsIntoGenerateShapeOpPass::InitializePatterns(
    pir::IrContext* context) {
  pir::RewritePatternSet ps(context);
  // elementwise ops
  ps.Add<FuseShapeOpsIntoGenerateShapeOpPattern<paddle::dialect::ExpandOp>>(context, ShapeOrDataDimExprs4Value_);
  
  return ps;
}

bool FuseShapeOpsIntoGenerateShapeOpPass::CanApplyOn(pir::Operation* op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
