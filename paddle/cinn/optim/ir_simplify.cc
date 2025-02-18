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

#include "paddle/cinn/optim/ir_simplify.h"

#include <absl/container/flat_hash_map.h>
#include <ginac/ginac.h>
#include <glog/logging.h>

#include <map>
#include <string>

#include "paddle/cinn/common/arithmetic.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT
using cinn::common::bfloat16;
using cinn::common::ExprToGinacConverter;
using cinn::common::float16;
using utils::GetStreamCnt;
using utils::Replace;

namespace {

//! Simplify the expression but Load.
struct SimplifyNoPureMathMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

#define __(op__)                                    \
  void Visit(const op__* op, Expr* expr) override { \
    *expr = ArithSimplify(*expr);                   \
  }

  __(Add)
  __(Mul)
  __(Sub)
  __(Div)
  __(Min)
  __(Max)
#undef __
};

struct SimplifyLoadMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Load* expr, Expr* op) override {
    auto* node = op->As<Load>();
    for (auto& idx : node->indices) {
      if (cinn::common::IsPureMath(idx)) {
        idx = ArithSimplify(idx);
      } else {
        SimplifyNoPureMathMutator()(&idx);
      }
    }
  }

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<For>();
    operator()(&node->body);
    operator()(&node->extent);
  }
};

struct SimplifyStoreMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Store* expr, Expr* op) override {
    auto* node = op->As<Store>();

    for (auto& idx : node->indices) {
      if (cinn::common::IsPureMath(idx)) {
        idx = ArithSimplify(idx);
      } else {
        SimplifyNoPureMathMutator()(&idx);
      }
    }
  }

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<For>();
    operator()(&node->body);
    operator()(&node->extent);
  }
};

struct SimplifyRampMutator : public ir::IRMutator<Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Ramp* op, Expr* expr) override {
    auto* node = expr->As<ir::Ramp>();

    PADDLE_ENFORCE_EQ(
        cinn::common::IsPureMath(node->base),
        true,
        ::common::errors::InvalidArgument("node->base is not a pure math!"));
    PADDLE_ENFORCE_EQ(
        cinn::common::IsPureMath(node->stride),
        true,
        ::common::errors::InvalidArgument("node->stride is not a pure math!"));
    node->base = ArithSimplify(node->base);
    node->stride = ArithSimplify(node->stride);
  }
  // ramp + ramp
  void Visit(const Add* op, Expr* expr) override {
    auto* node = expr->As<ir::Add>();
    Expr a = node->a();
    Expr b = node->b();
    auto a_ramp = a.As<ir::Ramp>();
    auto b_ramp = b.As<ir::Ramp>();

    if (a_ramp && b_ramp && a_ramp->lanes == b_ramp->lanes) {
      Expr base_add = optim::ArithSimplify(a_ramp->base + b_ramp->base);
      Expr stride_add = optim::ArithSimplify(a_ramp->stride + b_ramp->stride);
      *expr = ir::Ramp::Make(base_add, stride_add, a_ramp->lanes);
    }
  }
};

struct SimplifyIfThenElseMutator : public ir::IRMutator<> {
  void operator()(Expr* x) { ir::IRMutator<>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

  void Visit(const IfThenElse* op, Expr* expr) override {
    auto* node = expr->As<ir::IfThenElse>();

    auto* condition_int = node->condition.As<ir::IntImm>();
    auto* condition_uint = node->condition.As<ir::UIntImm>();

    // not deterministic
    if (!condition_int && !condition_uint) {
      Visit(&node->true_case, &node->true_case);
      if (node->false_case.defined()) {
        Visit(&node->false_case, &node->false_case);
      }
      return;
    }

    bool value = condition_int ? condition_int->value : condition_uint->value;
    if (value) {
      *expr = op->true_case;
      Visit(expr, expr);
    } else if (op->false_case.defined()) {
      *expr = op->false_case;
      Visit(expr, expr);
    } else {
      *expr = ir::Block::Make({});
    }
  }
};

struct SimplifySelectMutator : public ir::IRMutator<> {
  void operator()(Expr* x) { ir::IRMutator<>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

  void Visit(const Select* op, Expr* expr) override {
    auto* node = expr->As<ir::Select>();

    auto* condition_int = node->condition.As<ir::IntImm>();
    auto* condition_uint = node->condition.As<ir::UIntImm>();

    // not deterministic
    if (!condition_int && !condition_uint) {
      Visit(&node->true_value, &node->true_value);
      Visit(&node->false_value, &node->false_value);
      return;
    }

    bool value = condition_int ? condition_int->value : condition_uint->value;
    if (value) {
      *expr = op->true_value;
      Visit(expr, expr);
    } else {
      *expr = op->false_value;
      Visit(expr, expr);
    }
  }
};

struct SimplifyLogicalMutator : public ir::ExprMutator<> {
  void operator()(Expr* expr) { ir::ExprMutator<>::Visit(expr, expr); }

#define DEFINE_VISIT_CMP_OP(OpType, Method)                         \
  void Visit(const ir::OpType* op, Expr* expr) override {           \
    VLOG(7) << "Begin Visit Cmp op: " << *expr;                     \
    auto* node = expr->As<ir::OpType>();                            \
    ir::ExprMutator<>::Visit(&node->a(), &node->a());               \
    ir::ExprMutator<>::Visit(&node->b(), &node->b());               \
    if (node->a().is_constant() && node->b().is_constant())         \
      if (node->a().get_constant() Method node->b().get_constant()) \
        *expr = Expr(true);                                         \
    VLOG(7) << "End Visit Cmp op: " << *expr;                       \
  }
  DEFINE_VISIT_CMP_OP(LE, <=)
  DEFINE_VISIT_CMP_OP(LT, <)
  DEFINE_VISIT_CMP_OP(GE, >=)
  DEFINE_VISIT_CMP_OP(GT, >)
  DEFINE_VISIT_CMP_OP(EQ, ==)
  DEFINE_VISIT_CMP_OP(NE, !=)

#undef DEFINE_VISIT_CMP_OP

  void Visit(const ir::And* op, Expr* expr) override {
    VLOG(7) << "Begin Visit And op: " << *expr;
    auto* node = expr->As<ir::And>();
    ir::ExprMutator<>::Visit(&node->a(), &node->a());
    if (common::IsZero(node->a())) {
      *expr = Expr(false);
      VLOG(7) << "End Visit And op: " << *expr;
      return;
    }
    ir::ExprMutator<>::Visit(&node->b(), &node->b());
    if (common::IsZero(node->b())) {
      VLOG(7) << "End Visit And op: " << *expr;
      *expr = Expr(false);
      return;
    }
    if (common::IsOne(node->a()) && common::IsOne(node->b()))
      *expr = Expr(true);
    VLOG(7) << "End Visit And op: " << *expr;
  }

  void Visit(const ir::Or* op, Expr* expr) override {
    VLOG(7) << "Begin Visit Or op: " << *expr;
    auto* node = expr->As<ir::Or>();
    ir::ExprMutator<>::Visit(&node->a(), &node->a());
    if (common::IsOne(node->a())) {
      *expr = Expr(true);
      VLOG(7) << "End visit Or op: " << *expr;
      return;
    }
    ir::ExprMutator<>::Visit(&node->b(), &node->b());
    if (common::IsOne(node->b())) {
      *expr = Expr(true);
      VLOG(7) << "End visit Or op: " << *expr;
      return;
    }
    if (common::IsZero(node->a()) && common::IsZero(node->b()))
      *expr = Expr(false);
    VLOG(7) << "End visit Or op: " << *expr;
  }

  void Visit(const ir::Not* op, Expr* expr) override {
    VLOG(7) << "Begin Visit Not op: " << *expr;
    auto* node = expr->As<ir::Not>();
    auto v = node->v();
    ir::ExprMutator<>::Visit(&v, &v);
    switch (v.node_type()) {
      case ir::IrNodeTy::IntImm:
      case ir::IrNodeTy::UIntImm:
        *expr = common::IsZero(v) ? Expr(true) : Expr(false);
        return;
      case ir::IrNodeTy::Not:
        *expr = v.As<ir::Not>()->v();
        return;
      case ir::IrNodeTy::LE:
        *expr = ir::GT::Make(v->operand(0), v->operand(1));
        return;
      case ir::IrNodeTy::LT:
        *expr = ir::GE::Make(v->operand(0), v->operand(1));
        return;
      case ir::IrNodeTy::GE:
        *expr = ir::LT::Make(v->operand(0), v->operand(1));
        return;
      case ir::IrNodeTy::GT:
        *expr = ir::LE::Make(v->operand(0), v->operand(1));
        return;
      default:
        VLOG(7) << "End Visit Not op: " << *expr;
        return;
    }
    VLOG(7) << "End Visit Not op: " << *expr;
  }
};

struct ReplaceFracWithDivMutator : public ir::IRMutator<> {
  void operator()(Expr* x) { ir::IRMutator<>::Visit(x, x); }

  void Visit(const FracOp* op, Expr* expr) override {
    auto* node = expr->As<ir::FracOp>();

    ir::IRMutator<>::Visit(&node->operand(0), &node->operand(0));
    ir::IRMutator<>::Visit(&node->operand(1), &node->operand(1));

    *expr = ir::Div::Make(node->operand(0), node->operand(1));
  }
};

struct SimplifyBlocksMutator : public ir::IRMutator<> {
  SimplifyBlocksMutator() {}

  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

  void Visit(const Block* op, Expr* expr) override {
    auto* node = expr->As<ir::Block>();

    if (node->stmts.size() == 1 && node->stmts[0].As<ir::Block>()) {
      VLOG(6) << "Simplify size-1 ir::Block";
      *expr = node->stmts[0];
      Visit(expr, expr);
    } else {
      for (auto& s : node->stmts) {
        Visit(&s, &s);
      }
      std::vector<Expr> stmts;
      for (auto& s : node->stmts) {
        if (s.As<ir::Block>()) {
          VLOG(6) << "Simplify ir::Block inside ir::Block";
          auto inner_block = s.As<ir::Block>();
          for (auto inner_stmt : inner_block->stmts) {
            stmts.push_back(inner_stmt);
          }
        } else {
          stmts.push_back(s);
        }
      }
      expr->As<ir::Block>()->stmts = stmts;
    }
  }

  void Visit(const ScheduleBlock* op, Expr* expr) override {
    auto* node = expr->As<ScheduleBlock>();
    PADDLE_ENFORCE_NOT_NULL(node,
                            ::common::errors::InvalidArgument(
                                "The node expr->As<ScheduleBlock>() is null"));
    for (auto& var : node->iter_vars) {
      if (var->lower_bound.defined()) {
        Visit(&var->lower_bound, &var->lower_bound);
      }
      if (var->upper_bound.defined()) {
        Visit(&var->upper_bound, &var->upper_bound);
      }
    }
    for (auto& buffer_region : node->read_buffers) {
      Visit(&buffer_region, &buffer_region);
    }
    for (auto& buffer_region : node->write_buffers) {
      Visit(&buffer_region, &buffer_region);
    }

    if (node->body.As<Block>()) {
      if (node->body.As<Block>()->stmts.size() == 1) {
        node->body = node->body.As<Block>()->stmts[0];
      }
    }

    Visit(&(node->body), &(node->body));
  }
};

struct SimplifyForLoopsMutator : public ir::IRMutator<> {
  absl::flat_hash_map<std::string, Expr> var_mins;
  SimplifyForLoopsMutator() {}

  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    Visit(&node->min, &node->min);
    Visit(&node->extent, &node->extent);
    auto* min_i = node->min.As<IntImm>();
    auto* extent_i = node->extent.As<IntImm>();
    if (min_i && extent_i && extent_i->value - min_i->value == 1) {
      VLOG(6) << "Simplify current For Loop";
      std::string var_name = node->loop_var->name;
      var_mins.emplace(var_name, node->min);

      *expr = node->body;

      Visit(expr, expr);
      var_mins.erase(var_name);
    } else {
      Visit(&node->body, &node->body);
    }
  }

  void Visit(const _Var_* op, Expr* expr) override {
    auto* node = expr->As<ir::_Var_>();

    if (var_mins.count(node->name)) {
      *expr = var_mins.at(node->name);
    }
  }
};

template <typename CastType, typename T>
CastType NormCastValue(T value) {
  if (type_of<CastType>().is_uint() || type_of<T>().is_uint()) {
    // not support uint
    return static_cast<CastType>(value);
  }

  if (std::isinf(value)) {
    if (CastType(value) == -std::numeric_limits<CastType>::infinity()) {
      return -std::numeric_limits<CastType>::infinity();
    }
    return std::numeric_limits<CastType>::infinity();
  } else if (std::isnan(value)) {
    return std::numeric_limits<CastType>::signaling_NaN();
  } else if (value >= static_cast<T>(std::numeric_limits<CastType>::max())) {
    return std::numeric_limits<CastType>::max();
  } else if (value <= static_cast<T>(std::numeric_limits<CastType>::lowest())) {
    return std::numeric_limits<CastType>::lowest();
  }
  return static_cast<CastType>(value);
}

struct SimplifyCastMutator : public ir::IRMutator<> {
  void operator()(Expr* expr) { ir::IRMutator<ir::Expr*>::Visit(expr, expr); }

  void Visit(const ir::Cast* op, Expr* expr) {
    auto* node = expr->As<ir::Cast>();

    ir::IRMutator<ir::Expr*>::Visit(&node->v(), &node->v());

    if (op->type() == op->v().type()) {
      *expr = op->v();
      return;
    }

#define __CAST_TO_TYPE(type__)                                          \
  if (auto* i = op->v().As<ir::IntImm>()) {                             \
    *expr = Expr(static_cast<type__>(i->value));                        \
  } else if (auto* f = op->v().As<ir::FloatImm>()) {                    \
    *expr = Expr(static_cast<type__>(NormCastValue<type__>(f->value))); \
  } else if (auto* u = op->v().As<ir::UIntImm>()) {                     \
    *expr = Expr(static_cast<type__>(u->value));                        \
  } else {                                                              \
    CINN_NOT_IMPLEMENTED                                                \
  }

    if (op->v().is_constant()) {
      if (op->type() == type_of<int8_t>()) {
        __CAST_TO_TYPE(int8_t)
      } else if (op->type() == type_of<int16_t>()) {
        __CAST_TO_TYPE(int16_t)
      } else if (op->type() == type_of<int32_t>()) {
        __CAST_TO_TYPE(int32_t)
      } else if (op->type() == type_of<int64_t>()) {
        __CAST_TO_TYPE(int64_t)
      } else if (op->type() == type_of<uint8_t>()) {
        __CAST_TO_TYPE(uint8_t)
      } else if (op->type() == type_of<uint16_t>()) {
        __CAST_TO_TYPE(uint16_t)
      } else if (op->type() == type_of<uint32_t>()) {
        __CAST_TO_TYPE(uint32_t)
      } else if (op->type() == type_of<uint64_t>()) {
        __CAST_TO_TYPE(uint64_t)
      } else if (op->type() == type_of<float>()) {
        __CAST_TO_TYPE(float)
      } else if (op->type() == type_of<double>()) {
        __CAST_TO_TYPE(double)
      } else if (op->type() == type_of<bool>()) {
        __CAST_TO_TYPE(bool)
      } else if (op->type() == type_of<uint32_t>()) {
        __CAST_TO_TYPE(uint32_t)
      } else if (op->type() == type_of<uint64_t>()) {
        __CAST_TO_TYPE(uint64_t)
      } else if (op->type() == type_of<bfloat16>()) {
        // Cannot simplify!!! pass
        __CAST_TO_TYPE(bfloat16)
      } else if (op->type() == type_of<float16>()) {
        // Cannot simplify!!! pass
        __CAST_TO_TYPE(float16)
      } else {
        CINN_NOT_IMPLEMENTED
      }
    }
#undef __CAST_TO_TYPE
  }
};

}  // namespace

void SimplifyCast(Expr* expr) { SimplifyCastMutator()(expr); }
void SimplifyForLoops(Expr* expr) { SimplifyForLoopsMutator()(expr); }
void SimplifyBlocks(Expr* expr) { SimplifyBlocksMutator()(expr); }

void SimplifyLogical(Expr* expr) { SimplifyLogicalMutator()(expr); }

Expr ArithSimplify(const Expr& u) {
  if (!u.is_index()) return u;
  auto copied = ir_utils::IRCopy(u);
  return copied.as_index().Normalize();
}

void Simplify(Expr* expr) {
  VLOG(6) << "Begin Simplify " << *expr;
  SimplifyNoPureMathMutator()(expr);
  SimplifyCastMutator()(expr);
  SimplifyRampMutator()(expr);
  SimplifyLoadMutator()(expr);
  SimplifyStoreMutator()(expr);
  SimplifyLogicalMutator()(expr);
  SimplifyIfThenElseMutator()(expr);
  SimplifySelectMutator()(expr);
  SimplifyNoPureMathMutator()(expr);

  ReplaceFracWithDivMutator()(expr);
  VLOG(6) << "End Simplify " << *expr;
}
}  // namespace optim
}  // namespace cinn
