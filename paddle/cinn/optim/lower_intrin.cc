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

#include "paddle/cinn/optim/lower_intrin.h"

#include <string>

#include "paddle/cinn/backends/llvm/llvm_intrin_rule.h"
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/intrinsic_ops.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/registry.h"

namespace cinn {
namespace optim {

template <typename T>
void LowerIntrinImpl(const T &, const Target &target, ir::Expr *expr) {
  // Do nothing.
}

void LowerIntrinImpl(common::X86Arch, const Target &target, ir::Expr *expr) {
  codegen::RegisterCpuIntrinRule();

  struct Mutator : ir::IRMutator<Expr *> {
    Target target;

    explicit Mutator(Target target) : target(target) {}

    void operator()(ir::Expr *expr) {
      ir::IRMutator<ir::Expr *>::Visit(expr, expr);
    }

    void Visit(const ir::Add *op, Expr *expr) override {
      auto *node = expr->As<ir::Add>();
      PADDLE_ENFORCE_NOT_NULL(node, "The node can not be treat as a Add node.");
      Expr ret;
      if (node->type().is_float()) {
        if (const ir::Mul *mul = node->b().As<ir::Mul>()) {
          ret = ir::Call::Make(node->type(),
                               "fma",
                               {mul->a(), mul->b(), node->a()},
                               {},
                               ir::CallType::Intrinsic);
        } else if (const ir::Mul *mul = node->a().As<ir::Mul>()) {
          ret = ir::Call::Make(node->type(),
                               "fma",
                               {mul->a(), mul->b(), node->b()},
                               {},
                               ir::CallType::Intrinsic);
        }
        if (ret.defined()) {
          ir::IRMutator<>::Visit(&ret, &ret);
          *expr = ret;
          return;
        }
      }
      ir::IRMutator<>::Visit(&node->a(), &node->a());
      ir::IRMutator<>::Visit(&node->b(), &node->b());
    }

    void Visit(const ir::Call *op, Expr *expr) override {
      auto *node = expr->As<ir::Call>();
      PADDLE_ENFORCE_NOT_NULL(node,
                              "The node can not be treat as a Call node.");
      LowerCpuIntrinsicOp(node, expr);
    }

    void LowerCpuIntrinsicOp(ir::Call *op, Expr *expr) {
      auto *node = expr->As<ir::Call>();
      if (kIntrinsicCalls.count(node->name)) {
        PADDLE_ENFORCE_EQ(
            !node->name.empty(),
            true,
            ::common::errors::InvalidArgument("The node name is empty."));
        auto *func_ptr = ir::Registry::Get("lower_cpu_intrinsic_" + node->name);
        PADDLE_ENFORCE_NOT_NULL(
            func_ptr,
            ::common::errors::InvalidArgument(
                "find no rule to lower cpu intrinsic for lower_cpu_intrinsic_" +
                node->name));
        Expr ret = (*func_ptr)(Expr(node));
        if (!ret.same_as(*expr)) {
          ir::IRMutator<>::Visit(&ret, &ret);
        }
        *expr = ret;
        return;
      }
      for (auto &expr : node->read_args) {
        ir::IRMutator<>::Visit(&expr, &expr);
      }
      for (auto &expr : node->write_args) {
        ir::IRMutator<>::Visit(&expr, &expr);
      }
    }
  };

  Mutator m(target);
  m(expr);
}

void LowerIntrinByArch(ir::Expr *expr, const Target &target) {
  return std::visit(
      [&](const auto &impl) { return LowerIntrinImpl(impl, target, expr); },
      target.arch.variant());
}

void LowerIntrin(ir::Expr *expr, Target target) {
  return LowerIntrinByArch(expr, target);
}

}  // namespace optim
}  // namespace cinn
