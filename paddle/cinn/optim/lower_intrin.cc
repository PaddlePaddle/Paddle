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

void LowerIntrin(Expr *e, Target target) {
  if (target.arch == Target::Arch::X86) {
    codegen::RegisterCpuIntrinRule();
  } else {
    return;
  }
  struct Mutator : ir::IRMutator<Expr *> {
    Target target;

    explicit Mutator(Target target) : target(target) {}

    void operator()(Expr *e) { ir::IRMutator<>::Visit(e, e); }

    void Visit(const ir::Add *op, Expr *expr) override {
      auto *node = expr->As<ir::Add>();
      CHECK(node);
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
      CHECK(node);
      LowerCpuintrinsicOp(node, expr);
    }

    void LowerCpuintrinsicOp(ir::Call *op, Expr *expr) {
      auto *node = expr->As<ir::Call>();
      if (kIntrinsicCalls.count(node->name)) {
        CHECK(!node->name.empty());
        auto *func_ptr = ir::Registry::Get("lower_cpu_intrinsic_" + node->name);
        CHECK(func_ptr) << "find no rule to lower cpu intrinsic for "
                        << "lower_cpu_intrinsic_" + node->name;
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
  m(e);
}

}  // namespace optim
}  // namespace cinn
