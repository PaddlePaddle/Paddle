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

#pragma once

#include <map>

#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace backends {

// borrowed from Halide and TVM.
struct ModularEntry {
  int base;
  int coeff;

  ModularEntry() = default;
  ModularEntry(int base, int coeff) : base(base), coeff(coeff) {}

  static ModularEntry everything() { return ModularEntry{0, 1}; }

  static ModularEntry Add(const ModularEntry& a, const ModularEntry& b);
};

ModularEntry EvalModular(const Expr& e,
                         const std::map<Var, ModularEntry>& mod_map);

}  // namespace backends
}  // namespace cinn
