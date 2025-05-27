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

#include <cstring>
#include <functional>
#include <optional>
#include <ostream>
#include <vector>
#include "paddle/ap/include/axpr/atomic.h"
#include "paddle/ap/include/axpr/constants.h"

namespace ap::axpr {

using SymbolImpl = std::variant<tVar<std::string>, builtin_symbol::Symbol>;

struct Symbol : public SymbolImpl {
  using SymbolImpl::SymbolImpl;
  ADT_DEFINE_VARIANT_METHODS(SymbolImpl);

  std::size_t GetHashValue() const {
    std::size_t hash_value = Match(
        [&](const tVar<std::string>& var) {
          return std::hash<std::string>()(var.value());
        },
        [&](const builtin_symbol::Symbol& symbol) {
          return symbol.GetHashValue();
        });
    return adt::hash_combine(hash_value, this->index());
  }

  std::string Name() const {
    return Match(
        [](const tVar<std::string>& var) -> std::string { return var.value(); },
        [](const builtin_symbol::Symbol& symbol) -> std::string {
          return symbol.Name();
        });
  }
};

struct CoreExpr;

template <>
struct ExprSymbolTrait<CoreExpr> {
  using symbol_type = Symbol;
};

// (outer_func (inner_func [args]))
template <typename T>
struct ComposedCallImpl {
  T outer_func;
  T inner_func;
  std::vector<T> args;

  bool operator==(const ComposedCallImpl& other) const {
    return (this->outer_func == other.outer_func) &&
           (this->inner_func == other.inner_func) && (this->args == other.args);
  }
};

template <typename T>
ADT_DEFINE_RC(ComposedCall, const ComposedCallImpl<T>);

template <typename Expr>
using ComposedCallAtomic = ComposedCall<Atomic<Expr>>;

// core expr
// expr := aexpr | (aexpr (aexpr [aexpr]))
using CoreExprBase =
    std::variant<Atomic<CoreExpr>, ComposedCallAtomic<CoreExpr>>;

struct CoreExpr : public CoreExprBase {
  using CoreExprBase::CoreExprBase;
  ADT_DEFINE_VARIANT_METHODS(CoreExprBase);

  std::string ToSExpression() const;
};

size_t GetHashValue(const CoreExpr& core_expr);
size_t GetHashValue(const ComposedCallAtomic<CoreExpr>& composed_call);
size_t GetHashValue(const Atomic<CoreExpr>& atomic);
size_t GetHashValue(const Lambda<CoreExpr>& lambda);

inline size_t GetHashValue(const CoreExpr& core_expr) {
  size_t hash_value =
      core_expr.Match([&](const auto& impl) { return GetHashValue(impl); });
  return adt::hash_combine(hash_value, core_expr.index());
}

inline size_t GetHashValue(const ComposedCallAtomic<CoreExpr>& composed_call) {
  size_t ret = 0;
  ret = adt::hash_combine(ret, GetHashValue(composed_call->outer_func));
  ret = adt::hash_combine(ret, GetHashValue(composed_call->inner_func));
  for (const auto& arg : composed_call->args) {
    ret = adt::hash_combine(ret, GetHashValue(arg));
  }
  return ret;
}

inline size_t GetHashValue(const Atomic<CoreExpr>& atomic) {
  size_t ret = atomic.Match(
      [](const adt::Nothing) -> size_t { return 0; },
      [](const Symbol& symbol) -> size_t { return symbol.GetHashValue(); },
      [](const bool val) -> size_t { return val; },
      [](const int64_t val) -> size_t { return val; },
      [](const double val) -> size_t {
        size_t res;
        std::memcpy(&res, &val, sizeof(double));
        return res;
      },
      [](const std::string& val) -> size_t {
        return std::hash<std::string>()(val);
      },
      [](const Lambda<CoreExpr>& lambda) -> size_t {
        return GetHashValue(lambda);
      });
  return adt::hash_combine(ret, atomic.index());
}

inline size_t GetHashValue(const Lambda<CoreExpr>& lambda) {
  size_t ret = 0;
  for (const auto& arg : lambda->args) {
    ret = adt::hash_combine(ret, std::hash<std::string>()(arg.value()));
  }
  return adt::hash_combine(ret, GetHashValue(lambda->body));
}

}  // namespace ap::axpr

namespace std {

inline std::ostream& operator<<(std::ostream& os,
                                const ap::axpr::CoreExpr& core_expr) {
  return os << core_expr.ToSExpression();
}

template <>
struct hash<ap::axpr::CoreExpr> {
  size_t operator()(const ap::axpr::CoreExpr& core_expr) const {
    return GetHashValue(core_expr);
  }
};

template <>
struct hash<ap::axpr::Lambda<ap::axpr::CoreExpr>> {
  size_t operator()(
      const ap::axpr::Lambda<ap::axpr::CoreExpr>& core_expr) const {
    return GetHashValue(core_expr);
  }
};

template <>
struct hash<ap::axpr::Atomic<ap::axpr::CoreExpr>> {
  size_t operator()(
      const ap::axpr::Atomic<ap::axpr::CoreExpr>& core_expr) const {
    return GetHashValue(core_expr);
  }
};

template <>
struct hash<ap::axpr::ComposedCallAtomic<ap::axpr::CoreExpr>> {
  size_t operator()(
      const ap::axpr::ComposedCallAtomic<ap::axpr::CoreExpr>& core_expr) const {
    return GetHashValue(core_expr);
  }
};

}  // namespace std
