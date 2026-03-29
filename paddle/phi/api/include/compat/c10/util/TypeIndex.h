// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
// #Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace c10 {
namespace util {

class type_index final {
 public:
  constexpr explicit type_index(uint64_t checksum = 0) : checksum_(checksum) {}

  constexpr uint64_t underlyingId() const noexcept { return checksum_; }

  friend constexpr bool operator==(type_index lhs, type_index rhs) noexcept {
    return lhs.checksum_ == rhs.checksum_;
  }
  friend constexpr bool operator!=(type_index lhs, type_index rhs) noexcept {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(type_index lhs, type_index rhs) noexcept {
    return lhs.checksum_ < rhs.checksum_;
  }

 private:
  uint64_t checksum_;
};

namespace detail {

constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

constexpr uint64_t fnv1a64(const char* data, size_t n) {
  uint64_t hash = kFnvOffsetBasis;
  for (size_t i = 0; i < n; ++i) {
    hash ^= static_cast<uint64_t>(static_cast<unsigned char>(data[i]));
    hash *= kFnvPrime;
  }
  return hash;
}

template <typename T>
constexpr std::string_view type_signature() {
#if defined(_MSC_VER) && !defined(__clang__)
  constexpr std::string_view sig = __FUNCSIG__;
#else
  constexpr std::string_view sig = __PRETTY_FUNCTION__;
#endif
  return sig;
}

template <typename T>
constexpr uint64_t type_index_impl() {
  constexpr std::string_view sig = type_signature<T>();
  return fnv1a64(sig.data(), sig.size());
}

}  // namespace detail

template <typename T>
constexpr type_index get_type_index() {
  return type_index(detail::type_index_impl<std::decay_t<T>>());
}

}  // namespace util
}  // namespace c10

namespace std {
template <>
struct hash<c10::util::type_index> {
  size_t operator()(c10::util::type_index v) const noexcept {
    return static_cast<size_t>(v.underlyingId());
  }
};
}  // namespace std
