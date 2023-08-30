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

#include "cinn/adt/adt.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/match_and_rewrite.h"

namespace cinn::adt::equation {

DEFINE_ADT_TAG(tPointer);
DEFINE_ADT_TAG(tIterVar);

using IterVar = tIterVar<UniqueId>;

DEFINE_ADT_UNION(Constant,
                 std::size_t,
                 tStride<UniqueId>,
                 tDim<UniqueId>,
                 Neg<Constant>,
                 Add<Constant, Constant>,
                 Mul<Constant, Constant>);

// Undefined = {}
struct Undefined final {};

// IndexDot T = DotValue [Constant] T
template <typename StrideT, typename T>
struct DotValue : public Tuple<List<StrideT>, T> {
  using Tuple<List<StrideT>, T>::Tuple;
};

template <typename T>
struct IndexDot final : public DotValue<Constant, T> {
  using DotValue<Constant, T>::DotValue;
};

// IndexUnDot T = UnDotValue [Constant] T
template <typename StrideT, typename T>
struct UnDotValue : public Tuple<List<StrideT>, T> {
  using Tuple<List<StrideT>, T>::Tuple;
};

template <typename T>
struct IndexUnDot final : public UnDotValue<Constant, T> {
  using UnDotValue<Constant, T>::UnDotValue;
};

// ConstantAdd T = Add T Constant
template <typename T>
struct ConstantAdd final : public Add<T, Constant> {
  using Add<T, Constant>::Add;
};

// ConstantDiv T = Div T Constant
template <typename T>
struct ConstantDiv final : public Div<T, Constant> {
  using Div<T, Constant>::Div;
};

// ConstantMod T = Mod T Constant
template <typename T>
struct ConstantMod final : public Mod<T, Constant> {
  using Mod<T, Constant>::Mod;
};

// ListGetItem T = (T, Constant)
template <typename T>
struct ListGetItem final : public Tuple<T, Constant> {
  using Tuple<T, Constant>::Tuple;
};

// PtrGetItem T = (tPointer UniqueId, T)
template <typename T>
struct PtrGetItem final : public Tuple<tPointer<UniqueId>, T> {
  using Tuple<tPointer<UniqueId>, T>::Tuple;
};

DEFINE_ADT_UNION(Value,
                 Undefined,
                 IterVar,
                 List<Value>,
                 IndexDot<Value>,
                 IndexUnDot<Value>,
                 ConstantAdd<Value>,
                 ConstantDiv<Value>,
                 ConstantMod<Value>,
                 ListGetItem<Value>,
                 PtrGetItem<Value>);

}  // namespace cinn::adt::equation

namespace cinn::adt {

template <>
struct MatchTrait<equation::Value, equation::Undefined> final {
  static constexpr int is_template = false;
};

template <>
struct MatchTrait<equation::Value, equation::IterVar> final {
  static constexpr int is_template = false;
};

template <typename T>
struct MatchTrait<equation::Value, List<T>> final {
  using base_type = List<equation::Value>;
  using arg0_type = T;

  static constexpr int is_template = false;

  template <typename MatchPredicatorT>
  static bool MatchChildren(const base_type& list,
                            const MatchPredicatorT& MatchPredicator) {
    for (const auto& value : *list) {
      if (!MatchPredicator(value)) {
        return false;
      }
    }
    return true;
  }
};

#define DEFINE_ADT_MATCH_TRAIT_EQUATION(name)                            \
  template <typename T>                                                  \
  struct MatchTrait<equation::Value, equation::name<T>> final {          \
    using base_type = equation::name<equation::Value>;                   \
    using arg0_type = T;                                                 \
                                                                         \
    static constexpr int is_template = true;                             \
                                                                         \
    template <typename MatchPredicatorT>                                 \
    static bool MatchChildren(const base_type& value,                    \
                              const MatchPredicatorT& MatchPredicator) { \
      return MatchPredicator(std::get<0>(value.tuple()));                \
    }                                                                    \
  };

DEFINE_ADT_MATCH_TRAIT_EQUATION(IndexDot);
DEFINE_ADT_MATCH_TRAIT_EQUATION(IndexUnDot);
DEFINE_ADT_MATCH_TRAIT_EQUATION(ConstantAdd);
DEFINE_ADT_MATCH_TRAIT_EQUATION(ConstantDiv);
DEFINE_ADT_MATCH_TRAIT_EQUATION(ConstantMod);
DEFINE_ADT_MATCH_TRAIT_EQUATION(ListGetItem);
DEFINE_ADT_MATCH_TRAIT_EQUATION(PtrGetItem);

}  // namespace cinn::adt
