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
struct Match;

template <typename ExprT>
struct Match</*is_leaf*/ true, ExprT> final {
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
struct Match</*is_leaf*/ false, ExprT> final {
  template <typename source_pattern_type>
  static bool Call(const ExprT& pattern_expr) {
    return pattern_expr.Visit([](auto&& impl) {
      using pattern_type =
          typename MatchTrait<ExprT, source_pattern_type>::base_type;
      if constexpr (std::is_same<std::decay_t<decltype(impl)>,
                                 pattern_type>::value) {
        using arg0_type =
            typename MatchTrait<ExprT, source_pattern_type>::arg0_type;
        static constexpr bool is_arg0_expr_type =
            std::is_same<std::decay_t<ExprT>, arg0_type>::value;
        static constexpr bool is_arg0_template =
            ExprMatchTrait<is_arg0_expr_type>::
                template match_trait_type<ExprT, arg0_type>::is_template;
        static constexpr bool is_arg0_leaf =
            is_arg0_expr_type || !is_arg0_template;
        return MatchTrait<ExprT, source_pattern_type>::MatchChildren(
            impl, &Match<is_arg0_leaf, ExprT>::template Call<arg0_type>);
      } else {
        return false;
      }
    });
  }
};

template <typename ExprT,
          typename source_pattern_type,
          typename T,
          typename CtxT>
ExprT MatchAndRewrite(const ExprT& expr, const CtxT& ctx) {
  static constexpr bool is_leaf =
      std::is_same<ExprT, source_pattern_type>::value;
  if (Match<is_leaf, ExprT>::template Call<source_pattern_type>(expr)) {
    return T().MatchAndRewrite(expr, ctx);
  } else {
    return expr;
  }
}

template <typename ExprT, typename source_pattern_type, typename T>
ExprT MatchAndRewrite(const ExprT& expr) {
  static constexpr bool is_leaf =
      std::is_same<ExprT, source_pattern_type>::value;
  if (Match<is_leaf, ExprT>::template Call<source_pattern_type>(expr)) {
    return T().MatchAndRewrite(expr);
  } else {
    return expr;
  }
}

}  // namespace detail

template <typename T, typename ExprT, typename CtxT>
ExprT MatchAndRewrite(const ExprT& expr, const CtxT& ctx) {
  using l0_type = typename T::source_pattern_type;
  return detail::
      MatchAndRewrite<ExprT, typename T::source_pattern_type, T, CtxT>(expr,
                                                                       ctx);
}

template <typename T, typename ExprT>
ExprT MatchAndRewrite(const ExprT& expr) {
  using l0_type = typename T::source_pattern_type;
  return detail::MatchAndRewrite<ExprT, typename T::source_pattern_type, T>(
      expr);
}

}  // namespace cinn::adt
