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

#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/common/ir_util.h"

namespace cinn::common {
using namespace symbol;  // NOLINT

namespace {

struct DimExprToIrExprVisitor {
  ir::Expr ConvertToIrExpr(const DimExpr& dim_expr) {
    return std::visit(*this, dim_expr.variant());
  }

  ir::Expr operator()(const int64_t& dim) { return ir::Expr(dim); }

  ir::Expr operator()(const std::string& dim_expr) {
    Var x = ir::_Var_::Make(ir::Expr(static_cast<int64_t>(0)),
                            ir::Expr(INT64_MAX),
                            dim_expr,
                            /* is_reduce  = */ false,
                            /* is_symbolic_constant = */ true);
    return x;
  }

  ir::Expr operator()(const Negative<DimExpr>& dim_expr) {
    const auto& [operand] = *dim_expr;
    return ir::Sub::Make(ir::Expr(std::int64_t(0)), ConvertToIrExpr(operand));
  }

  ir::Expr operator()(const Reciprocal<DimExpr>& dim_expr) {
    const auto& [operand] = *dim_expr;
    return ir::Div::Make(ir::Expr(std::int64_t(1)), ConvertToIrExpr(operand));
  }

  ir::Expr operator()(const Add<DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    if (operands->empty()) {
      return ir::Expr(std::int64_t(0));
    }
    ir::Expr sum = ConvertToIrExpr(operands->at(0));
    for (std::size_t i = 1; i < operands->size(); ++i) {
      sum = ir::Add::Make(sum, ConvertToIrExpr(operands->at(i)));
    }
    return sum;
  }

  ir::Expr operator()(const Mul<DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    if (operands->empty()) {
      return ir::Expr(std::int64_t(1));
    }
    ir::Expr product = ConvertToIrExpr(operands->at(0));
    for (std::size_t i = 1; i < operands->size(); ++i) {
      product = ir::Mul::Make(product, ConvertToIrExpr(operands->at(i)));
    }
    return product;
  }

  ir::Expr operator()(const Max<DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    CHECK(!operands->empty());
    ir::Expr max = ConvertToIrExpr(operands->at(0));
    for (std::size_t i = 1; i < operands->size(); ++i) {
      max = ir::Max::Make(max, ConvertToIrExpr(operands->at(i)));
    }
    return max;
  }

  ir::Expr operator()(const Min<DimExpr>& dim_expr) {
    const auto& [operands] = dim_expr;
    CHECK(!operands->empty());
    ir::Expr min = ConvertToIrExpr(operands->at(0));
    for (std::size_t i = 1; i < operands->size(); ++i) {
      min = ir::Min::Make(min, ConvertToIrExpr(operands->at(i)));
    }
    return min;
  }

  ir::Expr operator()(const Broadcast<DimExpr>& dim_expr) {
    LOG(FATAL)
        << "no support for converting from Broadcast<DimExpr> to ir::Expr";
  }
};

}  // namespace

ir::Expr DimExprConverter::ConvertToIrExpr(const DimExpr& dim_expr) const {
  return DimExprToIrExprVisitor().ConvertToIrExpr(dim_expr);
}

}  // namespace cinn::common
