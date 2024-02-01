// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/adt/dim_expr_match_trait.h"
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/match.h"

namespace cinn::adt {

template <>
struct MatchTrait<Value, Undefined> final {
  static constexpr int is_template = false;
};

template <>
struct MatchTrait<Value, Iterator> final {
  static constexpr int is_template = false;
};

template <>
struct MatchTrait<Value, DimExpr> final {
  static constexpr int is_template = false;
};

template <typename T>
struct MatchTrait<Value, List<T>> final {
  using base_type = List<Value>;

  static constexpr int is_template = true;

  template <template <typename, typename> class Matcher>
  static bool MatchChildren(const base_type& list) {
    for (const auto& value : *list) {
      if (!Matcher<T, Value>::Call(value)) {
        return false;
      }
    }
    return true;
  }
};

#define DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(name, type0, type1) \
  template <typename T0, typename T1>                                \
  struct MatchTrait<Value, name<T0, T1>> final {                     \
    using base_type = name<type0, type1>;                            \
                                                                     \
    static constexpr int is_template = true;                         \
                                                                     \
    template <template <typename, typename> class Matcher>           \
    static bool MatchChildren(const base_type& value) {              \
      return Matcher<T0, type0>::Call(std::get<0>(value.tuple())) && \
             Matcher<T1, type1>::Call(std::get<1>(value.tuple()));   \
    }                                                                \
  };

DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(ListGetItem, Value, DimExpr);
DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(BroadcastedIterator, Value, DimExpr);
DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(IndexDotValue, Value, List<DimExpr>);
DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(IndexUnDotValue, Value, List<DimExpr>);
#undef DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2

#define DEFINE_ADT_MATCH_TRAIT_EQUATION(name)                     \
  template <typename T>                                           \
  struct MatchTrait<Value, name<T>> final {                       \
    using base_type = name<Value>;                                \
                                                                  \
    static constexpr int is_template = true;                      \
                                                                  \
    template <template <typename, typename> class Matcher>        \
    static bool MatchChildren(const base_type& value) {           \
      return Matcher<T, Value>::Call(std::get<0>(value.tuple())); \
    }                                                             \
  };

DEFINE_ADT_MATCH_TRAIT_EQUATION(PtrGetItem);
#undef DEFINE_ADT_MATCH_TRAIT_EQUATION

}  // namespace cinn::adt
