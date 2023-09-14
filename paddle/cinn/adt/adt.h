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
  Union(const Union&) = default;
  Union(Union&&) = default;

  template <
      typename Arg,
      std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, Union>, bool> = true>
  explicit Union(Arg&& arg) : variant_(std::forward<Arg>(arg)) {}

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
  Tuple(const Tuple&) = default;
  Tuple(Tuple&&) = default;
  Tuple& operator=(const Tuple&) = default;
  Tuple& operator=(Tuple&&) = default;

  template <typename... Args>
  explicit Tuple(Args&&... args)
      : tuple_(
            std::make_shared<std::tuple<Ts...>>(std::forward<Args>(args)...)) {}

  const std::tuple<Ts...>& tuple() const { return *tuple_; }

  template <std::size_t I>
  const auto& Get() const {
    return std::get<I>(*tuple_);
  }

 protected:
  std::shared_ptr<std::tuple<Ts...>> tuple_;
};

template <typename T>
bool TupleEqual(const T& lhs, const T& rhs) {
  if (&lhs.tuple() == &rhs.tuple()) {
    return true;
  }
  return lhs.tuple() == rhs.tuple();
}

template <typename T>
class List final {
 public:
  List(const List&) = default;
  List(List&&) = default;
  List& operator=(const List&) = default;
  List& operator=(List&&) = default;

  explicit List() : vector_(std::make_shared<std::vector<T>>()) {}

  template <typename... Args>
  explicit List(Args&&... args)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector{std::forward<Args>(args)...})) {}

  std::vector<T>& operator*() const { return *vector_; }
  std::vector<T>* operator->() const { return vector_.get(); }

  const std::vector<T>& vector() const { return *vector_; }

  const auto& Get(std::size_t idx) const { return vector_->at(idx); }

 private:
  std::shared_ptr<std::vector<T>> vector_;
};

template <typename T>
bool ListEqual(const T& lhs, const T& rhs) {
  if (&*lhs == &*rhs) {
    return true;
  }
  return *lhs == *rhs;
}

template <typename T>
class Tagged {
 public:
  Tagged(const Tagged&) = default;
  Tagged(Tagged&&) = default;
  Tagged& operator=(const Tagged&) = default;
  Tagged& operator=(Tagged&&) = default;

  template <
      typename Arg,
      std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, Tagged>, bool> = true>
  explicit Tagged(Arg&& value) : value_(value) {}

  const T& value() const { return value_; }

 private:
  T value_;
};

template <typename T>
bool TagEqual(const T& lhs, const T& rhs) {
  if (&lhs == &rhs) {
    return true;
  }
  return lhs.value() == rhs.value();
}

#define DEFINE_ADT_TAG(name)                \
  template <typename T>                     \
  struct name : public Tagged<T> {          \
    using Tagged<T>::Tagged;                \
    name(const name&) = default;            \
    name(name&&) = default;                 \
    name& operator=(const name&) = default; \
    name& operator=(name&&) = default;      \
  };

#define DEFINE_ADT_UNION(class_name, ...)                                      \
  class class_name final {                                                     \
   public:                                                                     \
    class_name(const class_name&) = default;                                   \
    class_name(class_name&&) = default;                                        \
    class_name& operator=(const class_name& other) = default;                  \
    class_name& operator=(class_name&& other) = default;                       \
                                                                               \
    template <typename Arg,                                                    \
              std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, class_name>, \
                               bool> = true>                                   \
    class_name(Arg&& arg) : variant_(std::forward<Arg>(arg)) {}                \
                                                                               \
    template <typename T>                                                      \
    const T& Get() const {                                                     \
      return std::get<T>(variant_);                                            \
    }                                                                          \
                                                                               \
    template <typename T>                                                      \
    bool Has() const {                                                         \
      return std::holds_alternative<T>(variant_);                              \
    }                                                                          \
                                                                               \
    template <typename T>                                                      \
    auto Visit(const T& visitor) const {                                       \
      return std::visit(visitor, variant_);                                    \
    }                                                                          \
                                                                               \
    template <typename... Fs>                                                  \
    auto operator>>(match<Fs...> const& match) const {                         \
      return variant_ >> match;                                                \
    }                                                                          \
                                                                               \
    const std::variant<__VA_ARGS__>& variant() const { return variant_; }      \
                                                                               \
   private:                                                                    \
    std::variant<__VA_ARGS__> variant_;                                        \
  }

template <typename UnionT>
bool UnionEqual(const UnionT& lhs, const UnionT& rhs) {
  if (&lhs == &rhs) {
    return true;
  }
  return std::visit(
      [](auto&& lhs, auto&& rhs) {
        if constexpr (std::is_same<std::decay_t<decltype(lhs)>,
                                   std::decay_t<decltype(rhs)>>::value) {
          return lhs == rhs;
        } else {
          return false;
        }
      },
      lhs.variant(),
      rhs.variant());
}

#define DEFINE_ADT_UNARY(name)    \
  template <typename T>           \
  struct name : public Tuple<T> { \
    using Tuple<T>::Tuple;        \
  }

#define DEFINE_ADT_BINARY(name)        \
  template <typename T0, typename T1>  \
  struct name : public Tuple<T0, T1> { \
    using Tuple<T0, T1>::Tuple;        \
  }

DEFINE_ADT_UNARY(Neg);
DEFINE_ADT_BINARY(Add);
DEFINE_ADT_BINARY(Mul);
DEFINE_ADT_BINARY(Div);
DEFINE_ADT_BINARY(Mod);

#define OVERLOAD_OPERATOR_EQ_NE(type, function)              \
  inline bool operator==(const type& lhs, const type& rhs) { \
    return function(lhs, rhs);                               \
  }                                                          \
  inline bool operator!=(const type& lhs, const type& rhs) { \
    return !(lhs == rhs);                                    \
  }

template <typename T>
std::size_t TagHashValue(const T& tag) {
  return std::hash<std::decay_t<decltype(tag.value())>>()(tag.value());
}

#define OVERRIDE_TAG_GET_HASH_VALUE(cls) \
  inline std::size_t GetHashValue(const cls& tag) { return TagHashValue(tag); }

#define OVERRIDE_UNION_GET_HASH_VALUE(cls)                                 \
  inline std::size_t GetHashValue(const cls& union_obj) {                  \
    return std::visit([](const auto& impl) { return GetHashValue(impl); }, \
                      union_obj.variant());                                \
  }

using Name = std::string;

// Undefined = {}
struct Undefined final {
  bool operator==(const Undefined&) const { return true; }
  bool operator!=(const Undefined&) const { return false; }
};

// Ok = {}
struct Ok final {
  bool operator==(const Ok&) const { return true; }
  bool operator!=(const Ok&) const { return false; }
};

DEFINE_ADT_UNION(OpArgPos, Undefined, tIn<std::size_t>, tOut<std::size_t>);

#define ADT_TODO() LOG(FATAL) << "TODO"

inline std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
  return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
}

}  // namespace adt
}  // namespace cinn
