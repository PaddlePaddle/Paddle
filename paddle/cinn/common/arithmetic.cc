// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/arithmetic.h"

#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

using utils::GetStreamCnt;
using utils::Join;
using utils::Replace;
using utils::Split;
using namespace ir;  // NOLINT

#ifdef As
#undef As
#endif

std::string ExprToGinacConverter::Repr(const ir::Expr& expr) {
  auto* load_n = expr.As<Load>();
  auto* var_n = expr.As<_Var_>();
  auto* broadcast_n = expr.As<Broadcast>();
  auto* mod_n = expr.As<Mod>();
  auto* min_n = expr.As<Min>();
  auto* max_n = expr.As<Max>();
  auto* div_n = expr.As<Div>();
  auto* frac_n = expr.As<FracOp>();
  if (load_n || broadcast_n || mod_n || min_n || max_n || div_n || frac_n) {
    std::string repr = GetStreamCnt(expr);
    Replace(&repr, "[", "lsq_");
    Replace(&repr, "]", "_rsq");
    Replace(&repr, "(", "lb_");
    Replace(&repr, ")", "_rb");
    Replace(&repr, "+", "_add_");
    Replace(&repr, "-", "_sub_");
    Replace(&repr, ":", "_ref_");
    Replace(&repr, "*", "_mul_");
    Replace(&repr, "/", "_div_");
    // remove the spaces
    auto fields = utils::Split(repr, " ");
    repr = utils::Join(fields, "_");
    return repr;
  } else if (var_n) {
    return utils::GetStreamCnt(expr);
  }
  return "";
}

void ExprToGinacConverter::RecordExpr(const ir::Expr& expr) {
  repr_to_expr_[Repr(expr)] = expr;
}

GiNaC::ex ExprToGinacConverter::BuildHelper(ir::Expr expr) {
  auto* load_n = expr.As<Load>();
  auto* var_n = expr.As<_Var_>();
  auto* int_n = expr.As<IntImm>();
  auto* float_n = expr.As<FloatImm>();
  auto* add_n = expr.As<Add>();
  auto* sub_n = expr.As<Sub>();
  auto* mul_n = expr.As<Mul>();
  auto* div_n = expr.As<Div>();
  auto* minus_n = expr.As<Minus>();
  auto* broadcast_n = expr.As<Broadcast>();
  auto* mod_n = expr.As<Mod>();
  auto* frac_n = expr.As<FracOp>();
  auto* min_n = expr.As<Min>();
  auto* max_n = expr.As<Max>();

  bool is_integer_math = expr.type().is_int();

  bool is_invalid_arith =
      load_n || var_n || broadcast_n || mod_n || min_n || max_n;
  if (is_integer_math)
    is_invalid_arith = is_invalid_arith || div_n ||
                       frac_n;  // GiNac can't deal with integer division.

  if (is_invalid_arith) {
    RecordExpr(expr);
    std::string repr = Repr(expr);
    return CreateGinacSymbol(repr);
  } else if (int_n) {
    return int_n->value;
  } else if (float_n) {
    return float_n->value;
  } else if (add_n) {
    auto a = BuildHelper(add_n->a());
    auto b = BuildHelper(add_n->b());
    return (a + b) * 1;
  } else if (sub_n) {
    return (BuildHelper(sub_n->a()) - BuildHelper(sub_n->b()));
  } else if (mul_n) {
    return (BuildHelper(mul_n->a()) * BuildHelper(mul_n->b()));
  } else if (div_n) {
    return (BuildHelper(div_n->a()) / BuildHelper(div_n->b()));
  } else if (frac_n) {
    return (BuildHelper(frac_n->a()) / BuildHelper(frac_n->b()));
  } else if (minus_n) {
    return -BuildHelper(minus_n->v());
  } else {
    CINN_NOT_IMPLEMENTED
  }
}

GiNaC::ex ExprToGinacConverter::operator()(Expr expr) {
  // TODO(Superjomn) Replace this with cinn::common::IsPureMath(
  auto complex_nodes = ir::ir_utils::CollectIRNodes(expr, [](const Expr* n) {
    return n->As<Block>() ||    //
           n->As<PolyFor>() ||  //
           n->As<EQ>() ||       //
           n->As<NE>() ||       //
           n->As<LT>() ||       //
           n->As<LE>() ||       //
           n->As<GT>() ||       //
           n->As<GE>() ||       //
           n->As<And>() ||      //
           n->As<Or>() ||       //
           n->As<Not>() ||      //
           n->As<Let>() ||      //
           n->As<Call>() ||     //
           n->As<Select>() ||   //
           n->As<Store>() ||    //
           n->As<Alloc>() ||    //
           n->As<Free>() ||     //
           n->As<IfThenElse>();
  });

  PADDLE_ENFORCE_EQ(complex_nodes.empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "Ginac converter can only deal with simple math "
                        "expression, but get some complex nodes."));
  return BuildHelper(expr);
}

GiNaC::symbol ExprToGinacConverter::CreateGinacSymbol(const std::string& repr) {
  PADDLE_ENFORCE_EQ(
      !repr.empty(),
      true,
      ::common::errors::InvalidArgument("The repr should not be empty."));
  auto it = repr_to_ginac_.find(repr);
  if (it != repr_to_ginac_.end()) return it->second;

  GiNaC::symbol x(repr);
  repr_to_ginac_[repr] = x;
  return x;
}

GiNaC::symbol ExprToGinacConverter::CreateGinacSymbol(const ir::Expr& var) {
  PADDLE_ENFORCE_NOT_NULL(
      var.As<_Var_>(),
      ::common::errors::InvalidArgument("The var should not be nullptr."));
  return CreateGinacSymbol(Repr(var));
}

class GiNaCToExprVisitor : public GiNaC::symbol::visitor,
                           public GiNaC::numeric::visitor,
                           public GiNaC::add::visitor,
                           public GiNaC::mul::visitor,
                           public GiNaC::power::visitor,
                           public GiNaC::basic::visitor,
                           public GiNaC::visitor {
  std::map<std::string, ir::Expr>& repr_to_expr;
  ir::Expr cur;

 public:
  explicit GiNaCToExprVisitor(
      std::map<std::string, ir::Expr>& repr_to_expr)  // NOLINT
      : repr_to_expr(repr_to_expr) {}

  Expr operator()(GiNaC::ex ex) {
    ex.accept(*this);
    return cur;
  }

  void visit(const GiNaC::symbol& node) override {
    auto it = repr_to_expr.find(node.get_name());
    PADDLE_ENFORCE_NE(
        it,
        repr_to_expr.end(),
        ::common::errors::InvalidArgument("The node should be found."));
    cur = it->second;
  }

  void visit(const GiNaC::numeric& node) override {
    if (node.is_integer()) {
      cur = Expr(static_cast<int>(node.to_int()));
    } else {
      cur = Expr(static_cast<float>(node.to_double()));
    }
  }
  void visit(const GiNaC::add& node) override {
    node.op(0).accept(*this);
    Expr res = cur;

    for (int i = 1; i < node.nops(); i++) {
      node.op(i).accept(*this);
      res = res + cur;
    }

    cur = res;
  }

  void visit(const GiNaC::power& node) override {
    node.op(0).accept(*this);
    Expr a = cur;
    node.op(1).accept(*this);

    auto* intv = cur.As<IntImm>();
    PADDLE_ENFORCE_NOT_NULL(
        intv,
        ::common::errors::InvalidArgument("The intv should not be nullptr."));
    PADDLE_ENFORCE_EQ(
        intv->value,
        -1,
        ::common::errors::InvalidArgument("The power value should be -1."));

    cur = Div::Make(Expr(1), a);
  }

  void visit(const GiNaC::mul& node) override {
    node.op(0).accept(*this);
    Expr res = cur;

    for (int i = 1; i < node.nops(); i++) {
      node.op(i).accept(*this);
      res = res * cur;
    }

    cur = res;
  }
  void visit(const GiNaC::basic& basic) override { CINN_NOT_IMPLEMENTED }
};

Expr ExprToGinacConverter::GinacToExpr(const GiNaC::ex& ex) {
  GiNaCToExprVisitor visitor(repr_to_expr_);
  return visitor(ex);
}

bool IsPureMath(Expr expr) {
  std::set<IrNodeTy> valid_node_tys({
      IrNodeTy ::_Var_,
      IrNodeTy ::IntImm,
      IrNodeTy ::Sum,
      IrNodeTy ::Product,
      IrNodeTy ::FracOp,
      IrNodeTy ::FloatImm,
      IrNodeTy ::Add,
      IrNodeTy ::Sub,
      IrNodeTy ::Div,
      IrNodeTy ::Mul,
      IrNodeTy::Mod,
      IrNodeTy ::Minus,
  });

  auto complex_nodes = ir::ir_utils::CollectIRNodes(expr, [&](const Expr* n) {
    return !valid_node_tys.count(n->node_type());
  });
#ifdef CINN_DEBUG
  for (auto& node : complex_nodes) {
    VLOG(3) << "Found " << node->node_type() << " " << Expr(node);
  }
#endif
  return complex_nodes.empty();
}

bool MathContainsSymbol(Expr expr, Var symbol) {
  // Use diff(expr, x) and check the result is not zero.
  ExprToGinacConverter expr_converter;
  auto expr_ex = expr_converter(expr);
  if (!expr_converter.HasSymbol(symbol->name)) return false;
  return !ginac::diff(expr_ex, expr_converter.GetSymbol(symbol->name))
              .is_zero();
}

// lhs >= rhs.
std::tuple<Expr, bool /*positive*/> Solve(Expr lhs, Expr rhs, Var var) {
  static std::mutex ginac_mutex;
  std::lock_guard<std::mutex> guard(ginac_mutex);
  VLOG(4) << "Solve: " << lhs << "=" << rhs << " in " << var;
  ExprToGinacConverter converter;
  auto lhs_ex = converter(lhs);
  auto rhs_ex = converter(rhs);
  ginac::lst eqs{lhs_ex == rhs_ex};
  VLOG(4) << "eqs: " << eqs;
  const auto& symbol = converter.GetSymbol(var->name);
  ginac::lst vars{symbol};
  ginac::ex res = ginac::lsolve(eqs, vars);

  PADDLE_ENFORCE_EQ(
      res.nops(),
      1,
      ::common::errors::InvalidArgument("The res npos should be 1."));
  auto item = res.op(0);
  PADDLE_ENFORCE_EQ(
      item.nops(),
      2,
      ::common::errors::InvalidArgument("The item npos should be 2."));
  Expr value = converter.GinacToExpr(item.op(1));

  // tell the symbol
  auto diff = lhs_ex - rhs_ex;
  auto diff_res = ginac::diff(diff, symbol);
  PADDLE_ENFORCE_EQ(
      !diff_res.is_zero(),
      true,
      ::common::errors::InvalidArgument("The diff_res should not be zero."));
  return std::make_tuple(value, diff_res > 0);
}

bool MathIsZero(Expr expr) {
  if (!IsPureMath(expr)) return false;
  ExprToGinacConverter converter;

  auto ex = converter(expr);
  return ex.is_zero();
}

}  // namespace common
}  // namespace cinn
