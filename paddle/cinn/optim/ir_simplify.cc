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

#include <ginac/ginac.h>
#include <glog/logging.h>

#include <map>
#include <string>

#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/ir_visitor.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/optim/simplify_util.h"
#include "paddle/cinn/utils/string.h"
#include "paddle/utils/flat_hash_map.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT
using cinn::common::bfloat16;
using cinn::common::float16;
using utils::GetStreamCnt;
using utils::Replace;

namespace {

//! Simplify the expression but Load.
struct SimplifyNoPureMathMutator : public ir::IRMutator<ir::Expr*> {
  SimplifyNoPureMathMutator(
      ir::IndexExpr::OptLevel opt_level = ir::IndexExpr::OptLevel::kLevel1)
      : opt_level_(opt_level) {}
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

#define __(op__)                                    \
  void Visit(const op__* op, Expr* expr) override { \
    *expr = ArithSimplify(*expr, opt_level_);       \
  }

  __(Add)
  __(Mul)
  __(Sub)
  __(Div)
  __(Mod)
  __(Min)
  __(Max)
#undef __
  ir::IndexExpr::OptLevel opt_level_;
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

struct SimplifyRampMutator : public ir::IRMutator<Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Ramp* op, Expr* expr) override {
    auto* node = expr->As<ir::Ramp>();

    PADDLE_ENFORCE_EQ(
        IsPureMath(node->base),
        true,
        ::common::errors::InvalidArgument("node->base is not a pure math!"));
    PADDLE_ENFORCE_EQ(
        IsPureMath(node->stride),
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

struct SimplifyLoadStoreMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Load* expr, Expr* op) override {
    auto* node = op->As<Load>();
    for (auto& idx : node->indices) {
      idx = ArithSimplify(idx);
    }
  }

  void Visit(const Store* expr, Expr* op) override {
    auto* node = op->As<Store>();
    for (auto& idx : node->indices) {
      idx = ArithSimplify(idx);
    }
    ir::IRMutator<ir::Expr*>::Visit(&node->value, &node->value);
  }
};

struct SimplifyLogicalMutator : public ir::IRMutator<> {
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

#define DEFINE_VISIT_CMP_OP(OpType, Method)                           \
  void Visit(const ir::OpType* op, Expr* expr) override {             \
    VLOG(7) << "Begin Visit Cmp op: " << *expr;                       \
    auto* node = expr->As<ir::OpType>();                              \
    ir::IRMutator<>::Visit(&node->a(), &node->a());                   \
    ir::IRMutator<>::Visit(&node->b(), &node->b());                   \
    if (node->a().is_constant() && node->b().is_constant()) {         \
      if (node->a().get_constant() Method node->b().get_constant()) { \
        *expr = Expr(true);                                           \
      } else {                                                        \
        *expr = Expr(false);                                          \
      }                                                               \
    }                                                                 \
    VLOG(7) << "End Visit Cmp op: " << *expr;                         \
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
    ir::IRMutator<>::Visit(&node->a(), &node->a());
    if (common::IsZero(node->a())) {
      *expr = Expr(false);
      VLOG(7) << "End Visit And op: " << *expr;
      return;
    }
    ir::IRMutator<>::Visit(&node->b(), &node->b());
    if (common::IsZero(node->b())) {
      VLOG(7) << "End Visit And op: " << *expr;
      *expr = Expr(false);
      return;
    }
    if (common::IsOne(node->a()) && common::IsOne(node->b())) {
      *expr = Expr(true);
    } else if (common::IsOne(node->a())) {
      *expr = node->b();
    } else if (common::IsOne(node->b())) {
      *expr = node->a();
    }
    VLOG(7) << "End Visit And op: " << *expr;
  }

  void Visit(const ir::Or* op, Expr* expr) override {
    VLOG(7) << "Begin Visit Or op: " << *expr;
    auto* node = expr->As<ir::Or>();
    ir::IRMutator<>::Visit(&node->a(), &node->a());
    if (common::IsOne(node->a())) {
      *expr = Expr(true);
      VLOG(7) << "End visit Or op: " << *expr;
      return;
    }
    ir::IRMutator<>::Visit(&node->b(), &node->b());
    if (common::IsOne(node->b())) {
      *expr = Expr(true);
      VLOG(7) << "End visit Or op: " << *expr;
      return;
    }
    if (common::IsZero(node->a()) && common::IsZero(node->b())) {
      *expr = Expr(false);
    } else if (common::IsZero(node->a())) {
      *expr = node->b();
    } else if (common::IsZero(node->b())) {
      *expr = node->a();
    }
    VLOG(7) << "End visit Or op: " << *expr;
  }

  void Visit(const ir::Not* op, Expr* expr) override {
    VLOG(7) << "Begin Visit Not op: " << *expr;
    auto* node = expr->As<ir::Not>();
    auto v = node->v();
    ir::IRMutator<>::Visit(&v, &v);
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

struct SimplifyIfThenElseMutator : public ir::ExprMutator<> {
  void operator()(Expr* x) { ir::ExprMutator<>::Visit(x, x); }

  using ir::ExprMutator<>::Visit;

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

/*
Example 1:
  Select(a <= b, b, a) → max(a, b)
Example 2:
  Select(a <= b, a, b) → min(a, b)
Example 3:
  Select(a <= MAX, max(a, MIN), MAX) → min(max(a, MIN), MAX)
  Select(a <= MAX, max(MIN, a), MAX) → min(max(a, MIN), MAX)
Example 4:
  Select(MIN <= b, min(b, MAX), MIN) → max(min(b, MAX), MIN)
                                     → min(max(b, MIN), MAX)
  Select(MIN <= b, min(MAX, b), MIN) → max(min(b, MAX), MIN)
                                     → min(max(b, MIN), MAX)
*/
struct SimplifySelect2MinMaxMutator : public ir::ExprMutator<> {
  void operator()(Expr* x) { ir::ExprMutator<>::Visit(x, x); }

  using ir::ExprMutator<>::Visit;

  // Recursively optimize CompareOp operands
  template <typename T>
  void VisitCompare(T* op, Expr* expr) {
    Expr a = op->a();
    Expr b = op->b();
    ir::ExprMutator<>::Visit(&a, &a);
    ir::ExprMutator<>::Visit(&b, &b);

    if (a.get() != op->a().get() || b.get() != op->b().get()) {
      *expr = T::Make(a, b);
    }
  }

  void Visit(const ir::GE* op, Expr* expr) override { VisitCompare(op, expr); }
  void Visit(const ir::GT* op, Expr* expr) override { VisitCompare(op, expr); }
  void Visit(const ir::LE* op, Expr* expr) override { VisitCompare(op, expr); }
  void Visit(const ir::LT* op, Expr* expr) override { VisitCompare(op, expr); }

  void Visit(const Select* op, Expr* expr) override {
    auto* node = expr->As<ir::Select>();

    // 1. Recursively optimize sub-expressions
    Expr condition = node->condition;
    Expr true_value = node->true_value;
    Expr false_value = node->false_value;

    ir::ExprMutator<>::Visit(&condition, &condition);
    ir::ExprMutator<>::Visit(&true_value, &true_value);
    ir::ExprMutator<>::Visit(&false_value, &false_value);

    // 2. If sub-expressions are modified, rebuild the Select node
    if (condition.get() != node->condition.get() ||
        true_value.get() != node->true_value.get() ||
        false_value.get() != node->false_value.get()) {
      *expr = ir::Select::Make(condition, true_value, false_value);
      node = expr->As<ir::Select>();
    }

    // 3. Function to optimize Select into Min/Max when possible
    auto TryOptimizeSelect = [&](const Expr& a,
                                 const Expr& b,
                                 const Expr& x,
                                 const Expr& y) -> Expr {
      // Case 1: Select(a <= b, b, a) → max(a, b)
      if (x == b && y == a) {
        if (b.is_constant()) {
          return ir::Max::Make(a, b);
        } else {
          return ir::Max::Make(b, a);
        }
      }
      // Case 2: Select(a <= b, a, b) → min(a, b)
      if (x == a && y == b) {
        if (b.is_constant()) {
          return ir::Min::Make(a, b);
        } else {
          return ir::Min::Make(b, a);
        }
      }
      // Case 3: Select(a <= MAX, max(a, MIN), MAX) → min(max(a, MIN), MAX)
      if (auto* max = x.As<ir::Max>()) {
        if (max->a() == a) {
          if (max->b().is_constant() && y.is_constant() && b.is_constant()) {
            if (y.get_constant() == b.get_constant() &&
                (max->b()).get_constant() <= y.get_constant()) {
              return ir::Min::Make(ir::Max::Make(a, max->b()), b);
            }
          }
        } else if (max->b() == a) {
          // Select(a <= MAX, max(MIN, a), MAX) → min(max(a, MIN), MAX)
          if (max->a().is_constant() && y.is_constant() && b.is_constant()) {
            if (y.get_constant() == b.get_constant() &&
                (max->a()).get_constant() <= y.get_constant()) {
              return ir::Min::Make(ir::Max::Make(a, max->a()), b);
            }
          }
        }
      }
      // Case 4: Select(MIN <= b, min(b, Max), MIN) → max(min(b, MAX), MIN)
      //                                            → min(max(b, MIN), MAX)
      if (auto* min = x.As<ir::Min>()) {
        if (min->a() == b) {
          if ((min->b()).is_constant() && y.is_constant() && a.is_constant()) {
            if (y.get_constant() == a.get_constant() &&
                y.get_constant() <= (min->b()).get_constant()) {
              return ir::Min::Make(ir::Max::Make(b, a), min->b());
            }
          }
        } else if (min->b() == b) {
          // Select(MIN <= b, min(Max, b), MIN) → min(max(b, MIN), MAX)
          if ((min->a()).is_constant() && y.is_constant() && a.is_constant()) {
            if (y.get_constant() == a.get_constant() &&
                y.get_constant() <= (min->a()).get_constant()) {
              return ir::Min::Make(ir::Max::Make(b, a), min->a());
            }
          }
        }
      }
      return Expr(nullptr);
    };

    // 4. Try to optimize different comparison conditions by converting them to
    // <= logic
    if (auto* ge = node->condition.As<ir::GE>()) {
      // Select(a >= b, t, f) → Select(b <= a, t, f)
      Expr optimized = TryOptimizeSelect(
          ge->b(), ge->a(), node->true_value, node->false_value);
      if (optimized.defined()) {
        *expr = optimized;
        return;
      }
    } else if (auto* gt = node->condition.As<ir::GT>()) {
      // Select(a > b, t, f) → Select(a <= b, f, t)
      Expr optimized = TryOptimizeSelect(
          gt->a(), gt->b(), node->false_value, node->true_value);
      if (optimized.defined()) {
        *expr = optimized;
        return;
      }
    } else if (auto* le = node->condition.As<ir::LE>()) {
      // Select(a <= b, t, f) → Select(a <= b, t, f)
      Expr optimized = TryOptimizeSelect(
          le->a(), le->b(), node->true_value, node->false_value);
      if (optimized.defined()) {
        *expr = optimized;
        return;
      }
    } else if (auto* lt = node->condition.As<ir::LT>()) {
      // Select(a < b, t, f) → Select(b <= a, f, t)
      Expr optimized = TryOptimizeSelect(
          lt->b(), lt->a(), node->false_value, node->true_value);
      if (optimized.defined()) {
        *expr = optimized;
        return;
      }
    }
  }
};

// Optimizes pow(2.0f, ceil(log2(x))) pattern into more efficient bit
// manipulation:
// Original: pow(2.0f, ceil(log2(x)))
// Optimized: ldexpf(1.0f, exponent) where exponent is calculated via:
//   1. float_as_uint(x) - reinterpret float as uint32
//   2. right_shift(bits, 23) - extract exponent field
//   3. (exponent_raw & 0xFF) - 127 - adjust IEEE754 bias
//   4. +1 if mantissa is non-zero (for ceil behavior)
struct SimplifyPowerCeilLog2BitOpLdexpfMutator : public ir::ExprMutator<> {
  void operator()(Expr* expr) { ir::ExprMutator<>::Visit(expr, expr); }

  using ir::ExprMutator<>::Visit;
  void Visit(const ir::Call* op, Expr* expr) override {
    /// 1. First recursively process all sub-expressions
    std::vector<Expr> new_args;
    for (const auto& arg : op->read_args) {
      Expr new_arg = arg;
      Visit(&new_arg, &new_arg);
      new_args.push_back(new_arg);
    }

    // 2. Match target pattern: pow(base, ceil(log2(x)))
    if (op->name == "pow" && new_args.size() == 2) {
      const Expr& base = new_args[0];
      const Expr& exponent = new_args[1];

      // Check if exponent is ceil(log2(x))
      if (const ir::Call* ceil_call = exponent.As<ir::Call>()) {
        if (ceil_call->name == "ceil" && ceil_call->read_args.size() == 1) {
          if (const ir::Call* log2_call =
                  ceil_call->read_args[0].As<ir::Call>()) {
            if (log2_call->name == "log2" && log2_call->read_args.size() == 1 &&
                log2_call->read_args[0].type().is_float(32)) {
              /// Verify base is 2.0f for optimization
              bool is_base_two = false;
              if (base.is_constant()) {
                if (base.get_constant() == 2.0f) {
                  is_base_two = true;
                }
              }
              if (is_base_two) {
                // 3. Replace with bit operations + ldexpf
                Expr x = log2_call->read_args[0];  // Extract log2's argument

                // Create bit operations to compute ceil(log2(x))
                // (1) Reinterpret float as 32-bit integer
                Expr bits = ir::Call::Make(common::Int(32),
                                           "__float_as_uint",
                                           {x},
                                           {},
                                           ir::CallType::Extern,
                                           ir::FunctionRef(),
                                           0,
                                           {});

                std::vector<cinn::ir::Expr> shift_r_args = {bits, ir::Expr(23)};
                Expr shift_r = ir::Call::Make(common::Int(32),
                                              "right_shift",
                                              shift_r_args,
                                              {},
                                              ir::CallType::Extern,
                                              ir::FunctionRef(),
                                              0,
                                              {});
                // (2) Extract exponent part: ((bits >> 23) & 0xFF) - 127
                std::vector<cinn::ir::Expr> bitwise_and_exp_args = {
                    shift_r, ir::Expr(0xFF)};
                Expr bitwise_and_exp = ir::Call::Make(common::Int(32),
                                                      "bitwise_and",
                                                      bitwise_and_exp_args,
                                                      {},
                                                      ir::CallType::Extern,
                                                      ir::FunctionRef(),
                                                      0,
                                                      {});
                Expr exponent_raw =
                    ir::Sub::Make(bitwise_and_exp, ir::Expr(127));
                // 3. Check if mantissa is non-zero (i.e., if exponent+1 is
                // needed)
                std::vector<cinn::ir::Expr> bitwise_and_tail_args = {
                    bits, ir::Expr(0x007FFFFF)};
                Expr bitwise_and_tail = ir::Call::Make(common::Int(32),
                                                       "bitwise_and",
                                                       bitwise_and_tail_args,
                                                       {},
                                                       ir::CallType::Extern,
                                                       ir::FunctionRef(),
                                                       0,
                                                       {});
                Expr mantissa_non_zero =
                    ir::NE::Make(bitwise_and_tail, ir::Expr(0));
                // (4) Check if it's a normal number (exponent != -127)
                Expr is_normal = ir::NE::Make(exponent_raw, ir::Expr(-127));
                // (5) If needed, exponent += 1
                Expr exponent_final = ir::Add::Make(
                    exponent_raw,
                    ir::Select::Make(
                        ir::And::Make(is_normal, mantissa_non_zero),
                        ir::Expr(1),
                        ir::Expr(0)));
                // (6) Create final expression: ldexpf(1.0f, exponent_final)
                Expr new_expr = ir::Call::Make(op->type(),
                                               "ldexpf",
                                               {ir::Expr(1.0f), exponent_final},
                                               {},
                                               ir::CallType::Extern,
                                               ir::FunctionRef(),
                                               0,
                                               {});
                *expr = new_expr;
                return;
              }
            }
          }
        }
      }
    }

    // For non-target patterns, reconstruct as-is
    if (new_args != op->read_args) {
      *expr = ir::Call::Make(op->type(),
                             op->name,
                             new_args,
                             op->write_args,
                             op->call_type,
                             op->func,
                             op->value_index,
                             op->attrs);
    }
  }
};

struct SimplifyUnitBlockMutator : public ir::ExprMutator<> {
  void operator()(Expr* x) { ir::ExprMutator<ir::Expr*>::Visit(x, x); }

  using ir::ExprMutator<>::Visit;

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

    if (node->body.As<Block>()) {
      if (node->body.As<Block>()->stmts.size() == 1) {
        node->body = node->body.As<Block>()->stmts[0];
      }
    }
    Visit(&(node->body), &(node->body));
  }
};

struct SimplifyUnitLoopMutator : public ir::IRMutator<> {
  paddle::flat_hash_map<std::string, Expr> var_mins;
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    Visit(&node->min, &node->min);
    Visit(&node->extent, &node->extent);
    auto* min_i = node->min.As<IntImm>();
    auto* extent_i = node->extent.As<IntImm>();
    if (min_i && extent_i && extent_i->value - min_i->value == 1) {
      VLOG(6) << "Simplify current Unit For Loop";
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
}  // namespace

void SimplifyCast(Expr* expr) { SimplifyCastMutator()(expr); }
void SimplifyUnitLoop(Expr* expr) { SimplifyUnitLoopMutator()(expr); }
void SimplifyUnitBlock(Expr* expr) { SimplifyUnitBlockMutator()(expr); }

void SimplifyLogical(Expr* expr) { SimplifyLogicalMutator()(expr); }
void SimplifyNoPureMath(Expr* expr, const ir::IndexExpr::OptLevel& opt_level) {
  auto mutator = SimplifyNoPureMathMutator(opt_level);
  mutator(expr);
}

Expr ArithSimplify(const Expr& u, const ir::IndexExpr::OptLevel& opt_level) {
  VLOG(3) << "Begin ArithSimplify " << u;
  if (!u.is_index()) return u;
  auto copied = ir_utils::IRCopy(u);
  auto res = copied.as_index().Normalize(opt_level);
  VLOG(3) << "End ArithSimplify " << res;
  return res;
}

void Simplify(Expr* expr) {
  VLOG(6) << "Begin Simplify " << *expr;
  ReplaceFracWithDivMutator()(expr);
  SimplifyNoPureMathMutator()(expr);
  SimplifyCastMutator()(expr);
  SimplifyRampMutator()(expr);
  SimplifyLoadStoreMutator()(expr);
  SimplifyLogicalMutator()(expr);
  SimplifyIfThenElseMutator()(expr);
  SimplifySelectMutator()(expr);
  SimplifySelect2MinMaxMutator()(expr);
  SimplifyPowerCeilLog2BitOpLdexpfMutator()(expr);
  SimplifyNoPureMathMutator()(expr);
  VLOG(6) << "End Simplify " << *expr;
}
}  // namespace optim
}  // namespace cinn
