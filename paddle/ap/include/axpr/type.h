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

#include <utility>
#include <variant>
#include "paddle/ap/include/axpr/adt.h"

namespace ap::axpr {

template <typename... Ts>
struct Type;

template <typename T>
struct TypeImpl {};

template <typename... Ts>
struct TypeImpl<Type<Ts...>> : public std::monostate {
  using value_type = Type<Ts...>;

  const char* Name() const { return "type"; }
};

template <typename TypeT, typename... Ts>
using TypeBase = std::variant<TypeImpl<TypeT>, TypeImpl<Ts>...>;

template <typename... Ts>
struct Type : public TypeBase<Type<Ts...>, Ts...> {
  using TypeBase<Type<Ts...>, Ts...>::TypeBase;

  ADT_DEFINE_VARIANT_METHODS(TypeBase<Type<Ts...>, Ts...>);

  std::string Name() const {
    return Match([](const auto& impl) -> std::string { return impl.Name(); });
  }
};

namespace detail {

template <typename T>
struct IsTypeHelper {
  static constexpr const bool value = false;
};

template <typename... Ts>
struct IsTypeHelper<Type<Ts...>> {
  static constexpr const bool value = true;
};

}  // namespace detail

template <typename T>
constexpr bool IsType() {
  return detail::IsTypeHelper<T>::value;
}

template <typename ValueT>
struct TypeTrait {
  using VariantT = std::decay_t<decltype(std::declval<ValueT>().variant())>;
  using TypeT = std::variant_alternative_t<0, VariantT>;
};

}  // namespace ap::axpr
