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

#include <type_traits>

namespace cinn::adt {
template <typename SumT, typename T>
struct MatchTrait;

namespace detail {

template <bool is_expr>
struct ExprMatchTrait;

template <>
struct ExprMatchTrait<true> final {
  template <typename ExprT, typename T>
  struct match_trait_type {
    static_assert(std::is_same<ExprT, T>::value, "");
    static constexpr int is_template = false;
  };
};

template <>
struct ExprMatchTrait<false> final {
  template <typename ExprT, typename T>
  using match_trait_type = MatchTrait<ExprT, T>;
};

template <bool is_leaf, typename ExprT>
struct DoMatch;

template <typename ExprT>
struct Match final {
  template <typename source_pattern_type>
  static bool Call(const ExprT& pattern_expr) {
    static constexpr bool is_expr =
        std::is_same<ExprT, source_pattern_type>::value;
    static constexpr bool is_template = ExprMatchTrait<is_expr>::
        template match_trait_type<ExprT, source_pattern_type>::is_template;
    static constexpr bool is_leaf = is_expr || !is_template;
    return DoMatch<is_leaf, ExprT>::template Call<source_pattern_type>(
        pattern_expr);
  }
};

template <typename ExprT>
struct DoMatch</*is_leaf*/ true, ExprT> final {
  template <typename source_pattern_type>
  static bool Call(const ExprT& pattern_expr) {
    if constexpr (std::is_same<std::decay_t<ExprT>,
                               source_pattern_type>::value) {
      return true;
    }
    return pattern_expr.Visit([](auto&& impl) {
      if constexpr (std::is_same<std::decay_t<decltype(impl)>,
                                 source_pattern_type>::value) {
        return true;
      } else {
        return false;
      }
    });
  }
};

template <typename ExprT>
struct DoMatch</*is_leaf*/ false, ExprT> final {
  template <typename source_pattern_type>
  static bool Call(const ExprT& pattern_expr) {
    return pattern_expr.Visit([](auto&& impl) {
      using pattern_type =
          typename MatchTrait<ExprT, source_pattern_type>::base_type;
      if constexpr (std::is_same<std::decay_t<decltype(impl)>,
                                 pattern_type>::value) {
        return MatchTrait<ExprT, source_pattern_type>::template MatchChildren<
            Match>(impl);
      } else {
        return false;
      }
    });
  }
};

}  // namespace detail

template <typename SourcePatternT, typename ExprT>
bool Match(const ExprT& expr) {
  return detail::Match<ExprT>::template Call<SourcePatternT>(expr);
}

}  // namespace cinn::adt
