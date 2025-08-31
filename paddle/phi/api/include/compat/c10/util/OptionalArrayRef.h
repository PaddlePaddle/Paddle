// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once
#include <c10/util/ArrayRef.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace c10 {
template <typename T>
class OptionalArrayRef final {
 public:
  // Constructors

  constexpr OptionalArrayRef() noexcept = default;

  constexpr OptionalArrayRef(std::nullopt_t) noexcept {}

  OptionalArrayRef(const OptionalArrayRef& other) = default;

  OptionalArrayRef(OptionalArrayRef&& other) noexcept = default;

  constexpr OptionalArrayRef(const std::optional<ArrayRef<T>>& other) noexcept
      : wrapped_opt_array_ref(other) {}

  constexpr OptionalArrayRef(std::optional<ArrayRef<T>>&& other) noexcept
      : wrapped_opt_array_ref(std::move(other)) {}

  constexpr OptionalArrayRef(const T& value) noexcept
      : wrapped_opt_array_ref(value) {}

  template <
      typename U = ArrayRef<T>,
      std::enable_if_t<!std::is_same_v<std::decay_t<U>, OptionalArrayRef> &&
                           !std::is_same_v<std::decay_t<U>, std::in_place_t> &&
                           std::is_constructible_v<ArrayRef<T>, U&&> &&
                           std::is_convertible_v<U&&, ArrayRef<T>> &&
                           !std::is_convertible_v<U&&, T>,
                       bool> = false>
  constexpr OptionalArrayRef(U&& value) noexcept(
      std::is_nothrow_constructible_v<ArrayRef<T>, U&&>)
      : wrapped_opt_array_ref(std::forward<U>(value)) {}

  template <
      typename U = ArrayRef<T>,
      std::enable_if_t<!std::is_same_v<std::decay_t<U>, OptionalArrayRef> &&
                           !std::is_same_v<std::decay_t<U>, std::in_place_t> &&
                           std::is_constructible_v<ArrayRef<T>, U&&> &&
                           !std::is_convertible_v<U&&, ArrayRef<T>>,
                       bool> = false>
  constexpr explicit OptionalArrayRef(U&& value) noexcept(
      std::is_nothrow_constructible_v<ArrayRef<T>, U&&>)
      : wrapped_opt_array_ref(std::forward<U>(value)) {}

  template <typename... Args>
  constexpr explicit OptionalArrayRef(std::in_place_t ip,
                                      Args&&... args) noexcept
      : wrapped_opt_array_ref(ip, std::forward<Args>(args)...) {}

  template <typename U, typename... Args>
  constexpr explicit OptionalArrayRef(std::in_place_t ip,
                                      std::initializer_list<U> il,
                                      Args&&... args)
      : wrapped_opt_array_ref(ip, il, std::forward<Args>(args)...) {}

  constexpr OptionalArrayRef(const std::initializer_list<T>& Vec)
      : wrapped_opt_array_ref(ArrayRef<T>(Vec)) {}

  // Destructor

  ~OptionalArrayRef() = default;

  // Assignment

  constexpr OptionalArrayRef& operator=(std::nullopt_t) noexcept {
    wrapped_opt_array_ref = std::nullopt;
    return *this;
  }

  OptionalArrayRef& operator=(const OptionalArrayRef& other) = default;

  OptionalArrayRef& operator=(OptionalArrayRef&& other) noexcept = default;

  constexpr OptionalArrayRef& operator=(
      const std::optional<ArrayRef<T>>& other) noexcept {
    wrapped_opt_array_ref = other;
    return *this;
  }

  constexpr OptionalArrayRef& operator=(
      std::optional<ArrayRef<T>>&& other) noexcept {
    wrapped_opt_array_ref = std::move(other);
    return *this;
  }

  template <typename U = ArrayRef<T>,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<U>, OptionalArrayRef> &&
                std::is_constructible_v<ArrayRef<T>, U&&> &&
                std::is_assignable_v<ArrayRef<T>&, U&&>>>
  constexpr OptionalArrayRef& operator=(U&& value) noexcept(
      std::is_nothrow_constructible_v<ArrayRef<T>, U&&>&&
          std::is_nothrow_assignable_v<ArrayRef<T>&, U&&>) {
    wrapped_opt_array_ref = std::forward<U>(value);
    return *this;
  }

  // Observers

  constexpr ArrayRef<T>* operator->() noexcept {
    return &wrapped_opt_array_ref.value();
  }

  constexpr const ArrayRef<T>* operator->() const noexcept {
    return &wrapped_opt_array_ref.value();
  }

  constexpr ArrayRef<T>& operator*() & noexcept {
    return wrapped_opt_array_ref.value();
  }

  constexpr const ArrayRef<T>& operator*() const& noexcept {
    return wrapped_opt_array_ref.value();
  }

  constexpr ArrayRef<T>&& operator*() && noexcept {
    return std::move(wrapped_opt_array_ref.value());
  }

  constexpr const ArrayRef<T>&& operator*() const&& noexcept {
    return std::move(wrapped_opt_array_ref.value());
  }

  constexpr explicit operator bool() const noexcept {
    return wrapped_opt_array_ref.has_value();
  }

  constexpr bool has_value() const noexcept {
    return wrapped_opt_array_ref.has_value();
  }

  constexpr ArrayRef<T>& value() & { return wrapped_opt_array_ref.value(); }

  constexpr const ArrayRef<T>& value() const& {
    return wrapped_opt_array_ref.value();
  }

  constexpr ArrayRef<T>&& value() && {
    return std::move(wrapped_opt_array_ref.value());
  }

  constexpr const ArrayRef<T>&& value() const&& {
    return std::move(wrapped_opt_array_ref.value());
  }

  template <typename U>
  constexpr std::enable_if_t<std::is_convertible_v<U&&, ArrayRef<T>>,
                             ArrayRef<T>>
  value_or(U&& default_value) const& {
    return wrapped_opt_array_ref.value_or(std::forward<U>(default_value));
  }

  template <typename U>
  constexpr std::enable_if_t<std::is_convertible_v<U&&, ArrayRef<T>>,
                             ArrayRef<T>>
  value_or(U&& default_value) && {
    return wrapped_opt_array_ref.value_or(std::forward<U>(default_value));
  }

  // Modifiers

  constexpr void swap(OptionalArrayRef& other) noexcept {
    std::swap(wrapped_opt_array_ref, other.wrapped_opt_array_ref);
  }

  constexpr void reset() noexcept { wrapped_opt_array_ref.reset(); }

  template <typename... Args>
  constexpr std::enable_if_t<std::is_constructible_v<ArrayRef<T>, Args&&...>,
                             ArrayRef<T>&>
  emplace(Args&&... args) noexcept(
      std::is_nothrow_constructible_v<ArrayRef<T>, Args&&...>) {
    return wrapped_opt_array_ref.emplace(std::forward<Args>(args)...);
  }

  template <typename U, typename... Args>
  constexpr ArrayRef<T>& emplace(std::initializer_list<U> il,
                                 Args&&... args) noexcept {
    return wrapped_opt_array_ref.emplace(il, std::forward<Args>(args)...);
  }

 private:
  std::optional<ArrayRef<T>> wrapped_opt_array_ref;
};

using OptionalIntArrayRef = OptionalArrayRef<int64_t>;

inline bool operator==(const OptionalIntArrayRef& a1,
                       const IntArrayRef& other) {
  if (!a1.has_value()) {
    return false;
  }
  return a1.value() == other;
}

inline bool operator==(const c10::IntArrayRef& a1,
                       const c10::OptionalIntArrayRef& a2) {
  return a2 == a1;
}

}  // namespace c10
namespace at {
using c10::OptionalIntArrayRef;
}  // namespace at

namespace torch {
using c10::OptionalIntArrayRef;
}  // namespace torch
