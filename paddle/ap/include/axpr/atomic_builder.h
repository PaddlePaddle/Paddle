// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/axpr/atomic.h"

namespace ap::axpr {

template <typename Expr>
class AtomicExprBuilder {
 public:
  AtomicExprBuilder() {}
  AtomicExprBuilder(const AtomicExprBuilder&) = delete;
  AtomicExprBuilder(AtomicExprBuilder&&) = delete;

  Atomic<Expr> Var(const std::string& name) {
    return Atomic<Expr>{tVar<std::string>{name}};
  }

  Atomic<Expr> Bool(bool c) { return Atomic<Expr>{c}; }

  Atomic<Expr> Int64(int64_t c) { return Atomic<Expr>{c}; }

  Atomic<Expr> Double(double c) { return Atomic<Expr>{c}; }

  Atomic<Expr> None() { return Atomic<Expr>{adt::Nothing{}}; }

  Atomic<Expr> String(const std::string& str) { return Atomic<Expr>{str}; }

  Atomic<Expr> Lambda(const std::vector<tVar<std::string>>& args,
                      const Expr& body) {
    return Atomic<Expr>{ap::axpr::Lambda<Expr>{args, body}};
  }

 private:
};

}  // namespace ap::axpr
