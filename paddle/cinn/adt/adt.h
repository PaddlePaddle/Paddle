// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "glog/logging.h"

namespace cinn {
namespace adt {

template <class... Ts>
struct match : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
match(Ts...) -> match<Ts...>;

template <typename... Ts, typename... Fs>
constexpr decltype(auto) operator>>(std::variant<Ts...> const& v,
                                    match<Fs...> const& match) {
  return std::visit(match, v);
}

template <typename... Ts>
class Union {
 public:
  template <typename... Args>
  explicit Union(Args&&... args) : variant_(std::forward<Args>(args)...) {}

  template <typename... Fs>
  auto operator>>(match<Fs...> const& match) const {
    return variant_ >> match;
  }

  const std::variant<Ts...>& variant() const { return variant_; }

 private:
  std::variant<Ts...> variant_;
};

template <typename... Ts>
class Tuple {
 public:
  template <typename... Args>
  explicit Tuple(Args&&... args)
      : tuple_(
            std::make_shared<std::tuple<Ts...>>(std::forward<Args>(args)...)) {}

  const std::tuple<Ts...>& tuple() const { return tuple_; }

 protected:
  std::shared_ptr<std::tuple<Ts...>> tuple_;
};

template <typename T>
class List final {
 public:
  List(const List&) = default;
  List(List&&) = default;

  template <typename... Args>
  explicit List(Args&&... args)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector{std::forward<Args>(args)...})) {}

  std::vector<T>& operator*() const { return *vector_; }
  std::vector<T>* operator->() const { return vector_.get(); }

 private:
  std::shared_ptr<std::vector<T>> vector_;
};

template <typename T>
class Tagged {
 public:
  template <typename ValueT>
  explicit Tagged(ValueT&& value) : value_(value) {}

  const T& value() const { return value_; }

 private:
  T value_;
};

#define DEFINE_ADT_TAG(name)       \
  template <typename T>            \
  struct name : public Tagged<T> { \
    using Tagged<T>::Tagged;       \
  };

#define DEFINE_ADT_UNION(class_name, ...)                            \
  struct class_name final : public ::cinn::adt::Union<__VA_ARGS__> { \
    using ::cinn::adt::Union<__VA_ARGS__>::Union;                    \
  };

using Name = std::string;

#define ADT_TODO() LOG(FATAL) << "TODO"

inline std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
  return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
}

}  // namespace adt
}  // namespace cinn
