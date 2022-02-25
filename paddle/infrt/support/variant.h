// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

// This file implements the variant data structure similar to
// absl::variant in C++17.

#pragma once

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <utility>

#include "paddle/infrt/support/type_traits.h"

namespace infrt {

// A Variant similar to absl::variant in C++17.
//
// Example usage:
//
// Variant<int, float, double> v;
//
// v = 1;
// assert(v.get<int>() == 1);
// assert(v.is<int>());
// assert(v.get_if<float>() == nullptr);
//
// // Print the variant.
// visit([](auto& t) { std::cout << t; }, v);
//
// v.emplace<float>(3);
//
template <typename... Ts>
class Variant {
  // Convenient constant to check if a type is a variant.
  template <typename T>
  static constexpr bool IsVariant =
      std::is_same<std::decay_t<T>, Variant>::value;

 public:
  using IndexT = int16_t;
  using Types = std::tuple<Ts...>;
  template <int N>
  using TypeOf = typename std::tuple_element<N, Types>::type;
  static constexpr size_t kNTypes = sizeof...(Ts);

  // Default constructor sets the Variant to the default constructed fisrt type.
  Variant() {
    using Type0 = TypeOf<0>;
    index_ = 0;
    new (&storage_) Type0();
  }

  template <typename T, std::enable_if_t<!IsVariant<T>, int> = 0>
  explicit Variant(T&& t) {
    fillValue(std::forward<T>(t));
  }

  Variant(const Variant& v) {
    visit([this](auto& t) { fillValue(t); }, v);
  }

  Variant(Variant&& v) {
    visit([this](auto&& t) { fillValue(std::move(t)); }, v);
  }

  ~Variant() { destroy(); }

  Variant& operator=(Variant&& v) {
    visit([this](auto& t) { *this = std::move(t); }, v);
    return *this;
  }

  Variant& operator=(const Variant& v) {
    visit([this](auto& t) { *this = t; }, v);
    return *this;
  }

  template <typename T, std::enable_if_t<!IsVariant<T>, int> = 0>
  Variant& operator=(T&& t) {
    destroy();
    fillValue(std::forward<T>(t));

    return *this;
  }

  template <typename T, typename... Args>
  T& emplace(Args&&... args) {
    AssertHasType<T>();

    destroy();
    index_ = IndexOf<T>;
    auto* t = new (&storage_) T(std::forward<Args>(args)...);
    return *t;
  }

  template <typename T>
  bool is() const {
    AssertHasType<T>();
    return IndexOf<T> == index_;
  }

  template <typename T>
  const T& get() const {
    AssertHasType<T>();
    return *reinterpret_cast<const T*>(&storage_);
  }

  template <typename T>
  T& get() {
    AssertHasType<T>();
    return *reinterpret_cast<T*>(&storage_);
  }

  template <typename T>
  const T* get_if() const {
    if (is<T>()) return &get<T>();
    return nullptr;
  }

  template <typename T>
  T* get_if() {
    if (is<T>()) return &get<T>();
    return nullptr;
  }

  IndexT index() const { return index_; }

  template <typename T>
  static constexpr size_t IndexOf = TupleIndexOf<T, Types>::value;

 private:
  static constexpr size_t kStorageSize = std::max({sizeof(Ts)...});
  static constexpr size_t kAlignment = std::max({alignof(Ts)...});

  template <typename T>
  static constexpr void AssertHasType() {
    constexpr bool has_type = TupleHasType<T, Types>::value;
    static_assert(has_type, "Invalid Type used for Variant");
  }

  void destroy() {
    visit(
        [](auto& t) {
          using T = std::decay_t<decltype(t)>;
          t.~T();
        },
        *this);
  }

  template <typename T>
  void fillValue(T&& t) {
    using Type = std::decay_t<T>;
    AssertHasType<Type>();

    index_ = IndexOf<Type>;
    new (&storage_) Type(std::forward<T>(t));
  }

  using StorageT = std::aligned_storage_t<kStorageSize, kAlignment>;

  StorageT storage_;
  IndexT index_ = -1;
};

struct Monostate {};

namespace internal {

template <typename F, typename Variant>
decltype(auto) visitHelper(
    F&& f,
    Variant&& v,
    std::integral_constant<int, std::decay_t<Variant>::kNTypes>) {
  assert(false && "Unexpected index_ in Variant");
}

// Disable clang-format as it does not format less-than (<) in the template
// parameter properly.
//
// clang-format off
template <
    typename F, typename Variant, int N,
    std::enable_if_t<N < std::decay_t<Variant>::kNTypes, int> = 0>
decltype(auto) visitHelper(F&& f, Variant&& v, std::integral_constant<int, N>) {
  // clang-format on
  using VariantT = std::decay_t<Variant>;
  using T = typename VariantT::template TypeOf<N>;
  if (auto* t = v.template get_if<T>()) {
    return f(*t);
  } else {
    return visitHelper(std::forward<F>(f),
                       std::forward<Variant>(v),
                       std::integral_constant<int, N + 1>());
  }
}

}  // namespace internal

template <typename F, typename Variant>
decltype(auto) visit(F&& f, Variant&& v) {
  return internal::visitHelper(std::forward<F>(f),
                               std::forward<Variant>(v),
                               std::integral_constant<int, 0>());
}

}  // namespace infrt
