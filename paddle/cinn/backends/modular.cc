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

#include "paddle/cinn/backends/modular.h"

#include "paddle/cinn/ir/ir_visitor.h"

namespace cinn {
namespace backends {

class ModularEvaluator : public ir::IRVisitorRequireReImpl<ModularEntry> {
 public:
  explicit ModularEvaluator(const std::map<Var, ModularEntry>& mod_map)
      : mod_map_(mod_map) {}

  ModularEntry Eval(const Expr& e) {
    return ir::IRVisitorRequireReImpl<ModularEntry>::Visit(&e);
  }

  ModularEntry Visit(const ir::IntImm* op) {
    if (op->value < std::numeric_limits<int>::max()) {
      return ModularEntry{static_cast<int>(op->value), 0};
    }
    return ModularEntry::everything();
  }

  ModularEntry Visit(const ir::UIntImm* op) {
    if (op->value < std::numeric_limits<uint64_t>::max()) {
      return ModularEntry{static_cast<int>(op->value), 0};
    }
    return ModularEntry::everything();
  }

  ModularEntry Visit(const ir::_Var_* op) {
    Var var(&Reference(op));
    auto it = mod_map_.find(var);
    if (it != mod_map_.end()) return it->second;
    return ModularEntry::everything();
  }

  ModularEntry Visit(const ir::Add* op) {
    auto a = Eval(op->a());
    auto b = Eval(op->b());
    ModularEntry ret;
    ret.coeff = gcd(a.coeff, b.coeff);
    ret.base = BaseSimplify(a.base + b.base, ret.coeff);
    return ret;
  }

  ModularEntry Visit(const ir::Sub* op) {
    auto a = Eval(op->a());
    auto b = Eval(op->b());

    ModularEntry ret;
    ret.coeff = gcd(a.coeff, b.coeff);
    ret.base = BaseSimplify(a.base - b.base, ret.coeff);
    return ret;
  }

  ModularEntry Visit(const ir::Mul* op) {
    auto a = Eval(op->a());
    auto b = Eval(op->b());

    int pq = a.coeff * b.coeff;
    int pm = a.coeff * b.base;
    int qn = a.base * b.coeff;

    ModularEntry ret;
    ret.coeff = gcd(pq, gcd(pm, qn));
    ret.base = BaseSimplify(a.base * b.base, ret.coeff);
    return ret;
  }

  ModularEntry Visit(const ir::Div* op) {
    auto a = Eval(op->a());
    auto b = Eval(op->b());

    if (b.coeff % b.base == 0) {
      ModularEntry ret;
      ret.coeff = a.coeff / b.base;
      ret.base = 0;
      return ret;
    }

    return ModularEntry::everything();
  }

  static int BaseSimplify(int base, int coeff) {
    if (coeff == 0) return base;
    base = base % coeff;
    if (base < 0) base += coeff;
    return base;
  }

  static int gcd(int a, int b) {
    CHECK_GE(a, 0);
    CHECK_GE(b, 0);
    if (a < b) std::swap(a, b);
    if (b == 0) return a;

    while (a % b != 0) {
      a = a % b;
      std::swap(a, b);
    }
    return b;
  }

 private:
  const std::map<Var, ModularEntry>& mod_map_;
};

ModularEntry ModularEntry::Add(const ModularEntry& a, const ModularEntry& b) {
  ModularEntry ret;
  ret.coeff = ModularEvaluator::gcd(a.coeff, b.coeff);
  ret.base = ModularEvaluator::BaseSimplify(a.base + b.base, ret.coeff);
  return ret;
}

}  // namespace backends
}  // namespace cinn
