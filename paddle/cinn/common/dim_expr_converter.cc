#include "paddle/cinn/common/dim_expr_converter.h"
#include "paddle/cinn/common/ir_util.h"

namespace cinn::common {
using namespace symbol;

namespace {

struct DimExprToIrExprVisitor {

  ir::Expr ConvertToIrExpr(const DimExpr& dim_expr) {
    return std::visit(*this, dim_expr);
  }

  ir::Expr operator()(const int64_t& dim) {
    return ir::Expr(dim);
  }

  ir::Expr operator()(const std::string& dim_expr) {
    Var x = ir::_Var_::Make(dim_expr, Int(64));
    return x;
  }

  ir::Expr operator()(const Negative<DimExpr>& dim_expr) {
    const auto& [operand] = *dim_expr;
    return ir::Sub::Make(ir::Expr(0), ConvertToIrExpr(operand));
  }

  ir::Expr operator()(const Reciprocal<DimExpr>& dim_expr) {
    const auto& [operand] = *dim_expr;
    return ir::Div::Make(ir::Expr(1), ConvertToIrExpr(operand));
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
    LOG(FATAL) << "no support for converting from Broadcast<DimExpr> to ir::Expr";
  }

};

}

ir::Expr DimExprConverter::ConvertToIrExpr(const DimExpr& dim_expr) const {
  return DimExprToIrExprVisitor().ConvertToIrExpr(dim_expr);
}

}