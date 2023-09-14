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

#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/match.h"

namespace cinn::adt {

template <>
struct MatchTrait<equation::Constant, std::int64_t> final {
  static constexpr int is_template = false;
};

template <>
struct MatchTrait<equation::Constant, equation::tStride<equation::UniqueId>>
    final {
  static constexpr int is_template = false;
};

template <>
struct MatchTrait<equation::Constant, equation::tDim<equation::UniqueId>>
    final {
  static constexpr int is_template = false;
};

template <typename T>
struct MatchTrait<equation::Constant, List<T>> final {
  using base_type = List<equation::Constant>;

  static constexpr int is_template = true;

  template <template <typename> class Matcher>
  static bool MatchChildren(const base_type& list) {
    for (const auto& value : *list) {
      if (!Matcher<equation::Constant>::template Call<T>(value)) {
        return false;
      }
    }
    return true;
  }
};

template <typename T>
struct MatchTrait<equation::Constant, Neg<T>> final {
  using base_type = Neg<equation::Constant>;

  static constexpr int is_template = true;

  template <template <typename> class Matcher>
  static bool MatchChildren(const base_type& constant) {
    return Matcher<equation::Constant>::template Call<T>(
        std::get<0>(constant.tuple()));
  }
};

template <typename T0, typename T1>
struct MatchTrait<equation::Constant, Add<T0, T1>> final {
  using base_type = Add<equation::Constant, equation::Constant>;

  static constexpr int is_template = true;

  template <template <typename> class Matcher>
  static bool MatchChildren(const base_type& constant) {
    return Matcher<equation::Constant>::template Call<T0>(
               std::get<0>(constant.tuple())) &&
           Matcher<equation::Constant>::template Call<T1>(
               std::get<1>(constant.tuple()));
  }
};

template <typename T0, typename T1>
struct MatchTrait<equation::Constant, Mul<T0, T1>> final {
  using base_type = Mul<equation::Constant, equation::Constant>;

  static constexpr int is_template = true;

  template <template <typename> class Matcher>
  static bool MatchChildren(const base_type& constant) {
    return Matcher<equation::Constant>::template Call<T0>(
               std::get<0>(constant.tuple())) &&
           Matcher<equation::Constant>::template Call<T1>(
               std::get<1>(constant.tuple()));
  }
};

template <>
struct MatchTrait<equation::Value, Undefined> final {
  static constexpr int is_template = false;
};

template <>
struct MatchTrait<equation::Value, equation::IterVar> final {
  static constexpr int is_template = false;
};

template <typename T>
struct MatchTrait<equation::Value, List<T>> final {
  using base_type = List<equation::Value>;

  static constexpr int is_template = false;

  template <template <typename> class Matcher>
  static bool MatchChildren(const base_type& list) {
    for (const auto& value : *list) {
      if (!Matcher<equation::Value>::template Call<T>(value)) {
        return false;
      }
    }
    return true;
  }
};

#define DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(name, type0, type1)    \
  template <typename T0, typename T1>                                   \
  struct MatchTrait<equation::Value, equation::name<T0, T1>> final {    \
    using base_type = equation::name<equation::type0, equation::type1>; \
                                                                        \
    static constexpr int is_template = true;                            \
                                                                        \
    template <template <typename> class Matcher>                        \
    static bool MatchChildren(const base_type& value) {                 \
      return Matcher<equation::type0>::template Call<T0>(               \
                 std::get<0>(value.tuple())) &&                         \
             Matcher<equation::type1>::template Call<T1>(               \
                 std::get<1>(value.tuple()));                           \
    }                                                                   \
  };

DEFINE_MATCH_TRAIT_VALUE_UNION_ARGSIZE_2(ListGetItem, Value, Constant);

#define DEFINE_ADT_MATCH_TRAIT_EQUATION(name)                   \
  template <typename T>                                         \
  struct MatchTrait<equation::Value, equation::name<T>> final { \
    using base_type = equation::name<equation::Value>;          \
                                                                \
    static constexpr int is_template = true;                    \
                                                                \
    template <template <typename> class Matcher>                \
    static bool MatchChildren(const base_type& value) {         \
      return Matcher<equation::Value>::template Call<T>(        \
          std::get<0>(value.tuple()));                          \
    }                                                           \
  };

DEFINE_ADT_MATCH_TRAIT_EQUATION(IndexDot);
DEFINE_ADT_MATCH_TRAIT_EQUATION(IndexUnDot);
DEFINE_ADT_MATCH_TRAIT_EQUATION(ConstantAdd);
DEFINE_ADT_MATCH_TRAIT_EQUATION(ConstantDiv);
DEFINE_ADT_MATCH_TRAIT_EQUATION(ConstantMod);
DEFINE_ADT_MATCH_TRAIT_EQUATION(PtrGetItem);

}  // namespace cinn::adt
