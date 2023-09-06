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

#include "paddle/cinn/common/arithmatic.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/ir/utils/ir_visitor.h"
#include "paddle/cinn/optim/cast_simplify.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT
using common::ExprToGinacConverter;
using utils::GetStreamCnt;
using utils::Replace;

namespace {

//! Simplify some sub-expression in the `expr`. Due to the simplify strategy
//! just fit several kinds of IR noedes, we partition the original expression to
//! several sub-expression those supported by simplify, and process each of
//! them.
void PartialSimplify(
    Expr* expr,
    const absl::flat_hash_map<std::string, common::CasInterval>& var_intervals =
        {}) {
  *expr = common::AutoSimplify(*expr, var_intervals);
}

//! Simplify the expression but Load.
struct SimplifyNoPureMathMutator : public ir::IRMutator<ir::Expr*> {
  common::cas_intervals_t& var_intervals;
  explicit SimplifyNoPureMathMutator(
      common::cas_intervals_t& var_intervals)  // NOLINT
      : var_intervals(var_intervals) {}

  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

#define __(op__)                                    \
  void Visit(const op__* op, Expr* expr) override { \
    PartialSimplify(expr, var_intervals);           \
  }

  __(Add)
  __(Mul)
  __(Sub)
  __(Div)
  __(Min)
  __(Max)
#undef __

  void Visit(const PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    node->condition = common::SolveInequality(op->condition, op->iterator);

    Visit(&node->body, &node->body);
  }

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    Visit(&node->min, &node->min);
    Visit(&node->extent, &node->extent);
    auto* min_i = op->min.As<IntImm>();
    auto* extent_i = op->extent.As<IntImm>();
    if (min_i && extent_i && extent_i->value > min_i->value) {
      var_intervals.emplace(
          op->loop_var->name,
          common::CasInterval{min_i->value, extent_i->value - 1});
    } else {
      var_intervals.emplace(op->loop_var->name,
                            common::CasInterval{op->min, op->extent - 1});
    }

    Visit(&node->body, &node->body);
    if (min_i && extent_i) {
      var_intervals.erase(op->loop_var->name);
    }
  }

  void Visit(const _Tensor_* op, Expr* expr) override {
    auto* node = expr->As<ir::_Tensor_>();

    for (auto& e : node->shape) {
      PartialSimplify(&e, var_intervals);
    }
    for (auto& e : node->domain) {
      PartialSimplify(&e, var_intervals);
    }
  }
};

struct SimplifyLoadMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Load* expr, Expr* op) override {
    auto* node = op->As<Load>();
    for (auto& idx : node->indices) {
      if (common::IsPureMath(idx)) {
        PartialSimplify(&idx, var_intervals_);
      } else {
        SimplifyNoPureMathMutator mutator(var_intervals_);
        mutator(&idx);
      }
    }
  }

  void Visit(const For* op, Expr* expr) override {
    auto* min_i = op->min.As<IntImm>();
    auto* extent_i = op->extent.As<IntImm>();
    if (min_i && extent_i && extent_i->value > min_i->value) {
      var_intervals_.emplace(
          op->loop_var->name,
          common::CasInterval{min_i->value, extent_i->value - 1});
    }

    auto* node = expr->As<For>();

    operator()(&node->body);
    operator()(&node->extent);

    if (min_i && extent_i) {
      var_intervals_.erase(op->loop_var->name);
    }
  }

  common::cas_intervals_t var_intervals_;
};

struct SimplifyStoreMutator : public ir::IRMutator<ir::Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Store* expr, Expr* op) override {
    auto* node = op->As<Store>();

    for (auto& idx : node->indices) {
      if (common::IsPureMath(idx)) {
        PartialSimplify(&idx, var_intervals_);
      } else {
        SimplifyNoPureMathMutator mutator(var_intervals_);
        mutator(&idx);
      }
    }
  }

  void Visit(const For* op, Expr* expr) override {
    auto* min_i = op->min.As<IntImm>();
    auto* extent_i = op->extent.As<IntImm>();
    if (min_i && extent_i) {
      var_intervals_.emplace(
          op->loop_var->name,
          common::CasInterval{min_i->value, extent_i->value - 1});
    }

    auto* node = expr->As<For>();

    operator()(&node->body);
    operator()(&node->extent);

    if (min_i && extent_i) {
      var_intervals_.erase(op->loop_var->name);
    }
  }

  common::cas_intervals_t var_intervals_;
};

struct SimplifyRampMutator : public ir::IRMutator<Expr*> {
  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  void Visit(const Ramp* op, Expr* expr) override {
    auto* node = expr->As<ir::Ramp>();

    CHECK(common::IsPureMath(node->base))
        << node->base << "is not a pure math!";
    CHECK(common::IsPureMath(node->stride))
        << node->stride << "is not a pure math!";

    PartialSimplify(&node->base);
    PartialSimplify(&node->stride);
  }
  // ramp + ramp
  void Visit(const Add* op, Expr* expr) override {
    auto* node = expr->As<ir::Add>();
    Expr a = node->a();
    Expr b = node->b();
    auto a_ramp = a.As<ir::Ramp>();
    auto b_ramp = b.As<ir::Ramp>();

    if (a_ramp && b_ramp && a_ramp->lanes == b_ramp->lanes) {
      Expr base_add = common::AutoSimplify(a_ramp->base + b_ramp->base);
      Expr stride_add = common::AutoSimplify(a_ramp->stride + b_ramp->stride);
      *expr = ir::Ramp::Make(base_add, stride_add, a_ramp->lanes);
    }
  }
};

struct SimplifyIfThenElseMutator : public ir::IRMutator<> {
  void operator()(Expr* x) { ir::IRMutator<>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

  void Visit(const IfThenElse* op, Expr* expr) override {
    auto* node = expr->As<ir::IfThenElse>();
    node->condition = common::AutoSimplify(node->condition);

    auto* condition_int = node->condition.As<ir::IntImm>();
    auto* condition_uint = node->condition.As<ir::UIntImm>();
    int64_t value;
    if (condition_int || condition_uint) {
      if (condition_int) {
        value = condition_int->value;
      } else {
        value = condition_uint->value;
      }
      if (value) {
        *expr = op->true_case;
      } else {
        if (op->false_case.defined()) {
          *expr = op->false_case;
        } else {
          // null condition
          *expr = ir::Block::Make({});
        }
      }
    }
    if (expr->As<ir::IfThenElse>()) {
      if (node->true_case.defined()) Visit(&node->true_case, &node->true_case);
      if (node->false_case.defined())
        Visit(&node->false_case, &node->false_case);
    }
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
};

struct SimplifyForLoopsMutator : public ir::IRMutator<> {
  absl::flat_hash_map<std::string, common::CasInterval> var_intervals;
  SimplifyForLoopsMutator() {}

  void operator()(Expr* x) { ir::IRMutator<ir::Expr*>::Visit(x, x); }

  using ir::IRMutator<>::Visit;

  void Visit(const For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    Visit(&node->min, &node->min);
    Visit(&node->extent, &node->extent);
    auto* min_i = node->min.As<IntImm>();
    auto* extent_i = node->extent.As<IntImm>();
    if (min_i && extent_i && extent_i->value > min_i->value &&
        extent_i->value - min_i->value == 1) {
      VLOG(6) << "Simplify current For Loop";
      std::string var_name = node->loop_var->name;
      var_intervals.emplace(
          var_name, common::CasInterval{min_i->value, extent_i->value - 1});

      *expr = node->body;

      Visit(expr, expr);
      var_intervals.erase(var_name);
    } else {
      Visit(&node->body, &node->body);
    }
  }

  void Visit(const _Var_* op, Expr* expr) override {
    auto* node = expr->As<ir::_Var_>();

    if (var_intervals.count(node->name)) {
      auto loop_range = var_intervals.at(node->name);
      *expr = Expr(loop_range.l);
    }
  }
};

}  // namespace

void Simplify(Expr* expr) {
  VLOG(3) << "Begin Simplify " << *expr;
  optim::CastSimplify(expr);
  SimplifyRampMutator()(expr);
  SimplifyLoadMutator()(expr);
  SimplifyStoreMutator()(expr);
  SimplifyIfThenElseMutator()(expr);

  common::cas_intervals_t var_intervals;
  SimplifyNoPureMathMutator mutator(var_intervals);
  mutator(expr);

  ReplaceFracWithDivMutator()(expr);
}

void SimplifyForLoops(Expr* expr) { SimplifyForLoopsMutator()(expr); }
void SimplifyBlocks(Expr* expr) { SimplifyBlocksMutator()(expr); }

}  // namespace optim
}  // namespace cinn
