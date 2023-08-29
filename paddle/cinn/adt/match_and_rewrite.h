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

#include <iostream>
#include <memory>
#include <tuple>
#include <type_traits>
#include <variant>

class Expr;

template <typename T>
struct Foo final {
  explicit Foo(T arg0) : arg0_(arg0) {}
  Foo(const Foo&) = default;
  Foo(Foo&&) = default;

  using arg0_type = T;
  using base_type = Foo<Expr>;

  template <std::size_t I>
  auto Get() const {
    if constexpr (I == 0) {
      return arg0_;
    }
  }

  T arg0_;
};

template <typename T>
struct Bar final {
  explicit Bar(T arg0) : arg0_(arg0) {}
  Bar(const Bar&) = default;
  Bar(Bar&&) = default;

  using arg0_type = T;
  using base_type = Bar<Expr>;

  template <std::size_t I>
  auto Get() const {
    if constexpr (I == 0) {
      return arg0_;
    }
  }

  T arg0_;
};

struct ExprNode;

class Expr final {
 public:
  Expr(const Expr&) = default;
  Expr(Expr&&) = default;
  Expr(const std::shared_ptr<ExprNode>& expr_node) : expr_node_(expr_node) {}

  ExprNode& operator*() const { return *expr_node_; }
  ExprNode* operator->() const { return expr_node_.get(); }

  std::shared_ptr<ExprNode> expr_node_;
};

template <typename... Args>
Expr MakeExpr(Args&&... args) {
  return std::make_shared<ExprNode>(std::forward<Args>(args)...);
}

template <typename T>
Expr Make(T x) {
  return MakeExpr(T(x));
}

template <template <typename> class T, typename... Args>
Expr Make(Args&&... args) {
  return MakeExpr(T{std::forward<Args>(args)...});
}

// Undot (Dot $0) -> $0
// ExprNode = Foo Expr | Bar Expr | int | float
struct ExprNode {
  ExprNode(const ExprNode&) = default;
  ExprNode(ExprNode&&) = default;

  template <typename... Args>
  explicit ExprNode(Args&&... args) : variant_(std::forward<Args>(args)...) {}

  template <typename T>
  auto Get() const {
    return std::get<T>(variant_);
  }

  template <typename T>
  auto Visit(const T& visitor) {
    return std::visit(visitor, variant_);
  }

  std::variant<Foo<Expr>, Bar<Expr>, int, float> variant_;
};

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
    static constexpr int ArgSize() { return 0; }
  };
};

template <>
struct ExprMatchTrait<false> final {
  template <typename ExprT, typename T>
  using match_trait_type = MatchTrait<ExprT, T>;
};

template <bool is_leaf, typename ExprT, typename T>
struct DoMatchAndRewrite;

template <typename ExprT, typename T>
struct DoMatchAndRewrite</*is_leaf*/ true, ExprT, T> final {
  template <typename source_pattern_type>
  static ExprT Call(const ExprT& ret_expr, const ExprT& pattern_expr) {
    if constexpr (std::is_same<std::decay_t<ExprT>,
                               source_pattern_type>::value) {
      return T().MatchAndRewrite(ret_expr);
    } else {
      return pattern_expr->Visit([&](auto&& impl) -> ExprT {
        if constexpr (std::is_same<std::decay_t<decltype(impl)>,
                                   source_pattern_type>::value) {
          return T().MatchAndRewrite(ret_expr);
        } else {
          return ret_expr;
        }
      });
    }
  }
};

template <typename ExprT, typename T>
struct DoMatchAndRewrite</*is_leaf*/ false, ExprT, T> final {
  template <typename source_pattern_type>
  static ExprT Call(const ExprT& ret_expr, const ExprT& pattern_expr) {
    return pattern_expr->Visit([&](auto&& impl) -> ExprT {
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
                template match_trait_type<ExprT, arg0_type>::ArgSize() > 0;
        static constexpr bool is_arg0_leaf =
            is_arg0_expr_type || !is_arg0_template;
        const ExprT& arg0 =
            MatchTrait<ExprT, source_pattern_type>::template Arg<0>(impl);
        return DoMatchAndRewrite<is_arg0_leaf, ExprT, T>::template Call<
            arg0_type>(ret_expr, arg0);
      } else {
        return ret_expr;
      }
    });
  }
};

template <typename ExprT, typename source_pattern_type, typename T>
ExprT MatchAndRewrite(const ExprT& expr) {
  static constexpr bool is_leaf =
      std::is_same<ExprT, source_pattern_type>::value;
  return DoMatchAndRewrite<is_leaf, ExprT, T>::template Call<
      source_pattern_type>(expr, expr);
}

}  // namespace detail

template <typename ExprT, typename T>
ExprT MatchAndRewrite(const ExprT& expr) {
  using l0_type = typename T::source_pattern_type;
  return detail::MatchAndRewrite<ExprT, typename T::source_pattern_type, T>(
      expr);
}

struct GetArg4Bar final {
  using source_pattern_type = Bar<Expr>;

  Expr MatchAndRewrite(const Expr& expr) const {
    return expr->Get<Bar<Expr>>().Get<0>();
  }
};

/*
struct SimplifyUndotDot final {
  using source_pattern_type = Undot<Dot<Expr>>;
  Expr MatchAndRewrite(const Expr& expr) const {
    return expr->Get<Undot<Expr>>().Get<0>()->Get<Dot<Expr>>().Get<0>();
  }
};

struct SimplifyListGetItem final {
  using source_pattern_type = ListGetItem<List<Expr>>;
  Expr MatchAndRewrite(const Expr& expr) const {
    const auto& get_item_struct = expr->Get<ListGetItem<Expr>>();
    return
get_item_struct.Get<0>()->Get<List<Expr>>().at(get_item_struct.index());
  }
};
*/

struct GetArg4BarFoo final {
  using source_pattern_type = Bar<Foo<Expr>>;

  Expr MatchAndRewrite(const Expr& expr) const {
    return expr->Get<Bar<Expr>>().Get<0>()->Get<Foo<Expr>>().Get<0>();
  }
};

struct GetArg4BarFooBar final {
  using source_pattern_type = Bar<Foo<Bar<Expr>>>;

  Expr MatchAndRewrite(const Expr& expr) const {
    return expr->Get<Bar<Expr>>()
        .Get<0>()
        ->Get<Foo<Expr>>()
        .Get<0>()
        ->Get<Bar<Expr>>()
        .Get<0>();
  }
};

struct GetArg4BarFooBarInt final {
  using source_pattern_type = Bar<Foo<Bar<int>>>;

  Expr MatchAndRewrite(const Expr& expr) const {
    return expr->Get<Bar<Expr>>()
        .Get<0>()
        ->Get<Foo<Expr>>()
        .Get<0>()
        ->Get<Bar<Expr>>()
        .Get<0>();
  }
};

template <>
struct MatchTrait<Expr, int> final {
  static constexpr int ArgSize() { return 0; }
};

template <>
struct MatchTrait<Expr, float> final {
  static constexpr int ArgSize() { return 0; }
};

template <typename T>
struct MatchTrait<Expr, Foo<T>> final {
  using base_type = Foo<Expr>;
  using arg0_type = T;

  static constexpr int ArgSize() { return 1; }

  template <int I>
  static Expr Arg(const Foo<Expr>& foo) {
    return foo.Get<0>();
  }
};

template <typename T>
struct MatchTrait<Expr, Bar<T>> final {
  using base_type = Bar<Expr>;
  using arg0_type = T;

  static constexpr int ArgSize() { return 1; }

  template <int I>
  static Expr Arg(const Bar<Expr>& bar) {
    return bar.Get<0>();
  }
};

int main() {
  {
    // Bar 33
    auto a = Make<Bar>(Make<float>(33));
    Expr b = MatchAndRewrite<Expr, GetArg4Bar>(a);
    std::cout << b->Get<float>() << std::endl;
  }
  {
    // Bar (Foo 66)
    auto a = Make<Bar>(Make<Foo>(Make<int>(66)));
    Expr b = MatchAndRewrite<Expr, GetArg4BarFoo>(a);
    std::cout << b->Get<int>() << std::endl;
  }
  {
    // Bar (Foo (Bar 99))
    auto a = Make<Bar>(Make<Foo>(Make<Bar>(Make<int>(99))));
    Expr b = MatchAndRewrite<Expr, GetArg4BarFooBar>(a);
    std::cout << b->Get<int>() << std::endl;
  }
  {
    // Bar (Foo (Bar 888))
    auto a = Make<Bar>(Make<Foo>(Make<Bar>(Make<int>(888))));
    Expr b = MatchAndRewrite<Expr, GetArg4BarFooBarInt>(a);
    std::cout << b->Get<int>() << std::endl;
  }
  return 0;
}
