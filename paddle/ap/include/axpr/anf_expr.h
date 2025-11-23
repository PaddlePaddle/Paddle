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

#include <functional>
#include <optional>
#include <vector>
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/atomic.h"

namespace ap::axpr {

template <typename Expr>
struct IfImpl {
  Atomic<Expr> cond;
  Expr true_expr;
  Expr false_expr;

  bool operator==(const IfImpl& other) const {
    return (this->cond == other.cond) &&
           (this->true_expr == other.false_expr) &&
           (this->false_expr == other.false_expr);
  }
};

template <typename Expr>
ADT_DEFINE_RC(If, const IfImpl<Expr>);

template <typename Expr>
using CombinedBase = std::variant<Call<Expr>, If<Expr>>;

template <typename Expr>
struct Combined : public CombinedBase<Expr> {
  using CombinedBase<Expr>::CombinedBase;
  ADT_DEFINE_VARIANT_METHODS(CombinedBase<Expr>);
};

template <typename Expr>
struct Bind {
  tVar<std::string> var;
  Combined<Expr> val;

  bool operator==(const Bind& other) const {
    return this->var == other.var && this->val == other.val;
  }
};

template <typename Expr>
struct LetImpl {
  std::vector<Bind<Expr>> bindings;
  Expr body;

  bool operator==(const LetImpl& other) const {
    return this->bindings == other.bindings && this->body == other.body;
  }
};

template <typename Expr>
ADT_DEFINE_RC(Let, const LetImpl<Expr>);

struct AnfExpr;

// expr := aexpr | cexpr | let [VAR cexpr] expr
// cexpr := (aexpr aexpr ...) | (If aexpr expr expr)
using AnfExprBase =
    std::variant<Atomic<AnfExpr>, Combined<AnfExpr>, Let<AnfExpr>>;

// A-norm form
struct AnfExpr : public AnfExprBase {
  using AnfExprBase::AnfExprBase;
  ADT_DEFINE_VARIANT_METHODS(AnfExprBase);

  static constexpr const char* kString() { return "str"; }
  static constexpr const char* kLambda() { return "lambda"; }
  static constexpr const char* kIf() { return "if"; }
  static constexpr const char* kLet() { return "__builtin_let__"; }

  std::string DumpToJsonString() const;
  std::string DumpToJsonString(int indent) const;
};

}  // namespace ap::axpr
