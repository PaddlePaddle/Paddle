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
#include <type_traits>
#include <vector>
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/common/overloaded.h"

namespace ap::axpr {

template <typename Expr>
struct LambdaImpl {
  std::vector<tVar<std::string>> args;
  Expr body;

  bool operator==(const LambdaImpl& other) const {
    return (this->args == other.args) && (this->body == other.body);
  }
};

template <typename Expr>
ADT_DEFINE_RC(Lambda, const LambdaImpl<Expr>);

// aexpr := Var | CONST | (lambda [VAR] expr)

template <typename Expr>
struct ExprSymbolTrait {
  using symbol_type = tVar<std::string>;
};

template <typename Expr>
using AtomicBase = std::variant<typename ExprSymbolTrait<Expr>::symbol_type,
                                adt::Nothing,
                                bool,
                                int64_t,
                                double,
                                std::string,
                                Lambda<Expr>>;

template <typename Expr>
struct Atomic : public AtomicBase<Expr> {
  using AtomicBase<Expr>::AtomicBase;
  ADT_DEFINE_VARIANT_METHODS(AtomicBase<Expr>);
};

template <typename Expr>
struct CallImpl {
  Atomic<Expr> func;
  std::vector<Atomic<Expr>> args;

  bool operator==(const CallImpl& other) const {
    return (this->func == other.func) && (this->args == other.args);
  }
};

template <typename Expr>
ADT_DEFINE_RC(Call, const CallImpl<Expr>);

}  // namespace ap::axpr
