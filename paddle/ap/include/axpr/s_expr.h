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
#include <ostream>
#include <vector>
#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/atomic.h"

namespace ap::axpr {

struct SExpr;

// (outer_func (inner_func [args]))
template <typename Expr>
struct SListImpl {
  std::vector<Expr> children;

  bool operator==(const SListImpl& other) const {
    return this->children == other.children;
  }
};

template <typename Expr>
ADT_DEFINE_RC(SList, const SListImpl<Expr>);

// s expression
// expr := aexpr | ([expr])
using SExprBase = std::variant<Atomic<SExpr>, SList<SExpr>>;

struct SExpr : public SExprBase {
  using SExprBase::SExprBase;
  ADT_DEFINE_VARIANT_METHODS(SExprBase);

  std::string ToSExpression() const;
};

}  // namespace ap::axpr

namespace std {

inline std::ostream& operator<<(std::ostream& os,
                                const ap::axpr::SExpr& core_expr) {
  return os << core_expr.ToSExpression();
}

}  // namespace std
