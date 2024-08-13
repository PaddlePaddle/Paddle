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

  template <typename __T>
  const __T& Get() const {
    return std::get<__T>(variant_);
  }

  template <typename __T>
  bool Has() const {
    return std::holds_alternative<__T>(variant_);
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
  std::tuple<Ts...>* mut_tuple() { return &*tuple_; }

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

  using value_type = T;

  explicit List() : vector_(std::make_shared<std::vector<T>>()) {}

  template <
      typename Arg,
      std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, List>, bool> = true>
  explicit List(Arg&& arg)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector<T>{std::forward<Arg>(arg)})) {}

  template <typename Arg0, typename Arg1, typename... Args>
  List(Arg0&& arg0, Arg1&& arg1, Args&&... args)
      : vector_(std::make_shared<std::vector<T>>(
            std::vector<T>{std::forward<Arg0>(arg0),
                           std::forward<Arg1>(arg1),
                           std::forward<Args>(args)...})) {}

  bool operator==(const List& other) const {
    if (&vector() == &other.vector()) {
      return true;
    }
    return vector() == other.vector();
  }

  bool operator!=(const List& other) const { return !(*this == other); }

  std::vector<T>& operator*() const { return *vector_; }
  std::vector<T>* operator->() const { return vector_.get(); }

  const std::vector<T>& vector() const { return *vector_; }

  const auto& Get(std::size_t idx) const { return vector_->at(idx); }

 private:
  std::shared_ptr<std::vector<T>> vector_;
};

#define DEFINE_ADT_TAG(TagName)                                             \
  template <typename T>                                                     \
  class TagName {                                                           \
   public:                                                                  \
    TagName() = default;                                                    \
    TagName(const TagName&) = default;                                      \
    TagName(TagName&&) = default;                                           \
    TagName& operator=(const TagName&) = default;                           \
    TagName& operator=(TagName&&) = default;                                \
                                                                            \
    bool operator==(const TagName& other) const {                           \
      return value_ == other.value();                                       \
    }                                                                       \
                                                                            \
    bool operator!=(const TagName& other) const {                           \
      return value_ != other.value();                                       \
    }                                                                       \
                                                                            \
    template <typename Arg,                                                 \
              std::enable_if_t<!std::is_same_v<std::decay_t<Arg>, TagName>, \
                               bool> = true>                                \
    explicit TagName(Arg&& value) : value_(value) {}                        \
                                                                            \
    const T& value() const { return value_; }                               \
                                                                            \
   private:                                                                 \
    T value_;                                                               \
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
    template <typename __T>                                                    \
    const __T& Get() const {                                                   \
      return std::get<__T>(variant_);                                          \
    }                                                                          \
                                                                               \
    template <typename __T>                                                    \
    bool Has() const {                                                         \
      return std::holds_alternative<__T>(variant_);                            \
    }                                                                          \
                                                                               \
    template <typename __T>                                                    \
    auto Visit(const __T& visitor) const {                                     \
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

#define OVERRIDE_UNION_GET_HASH_VALUE(cls)                                     \
  inline std::size_t GetHashValue(const cls& union_obj) {                      \
    return std::visit([](const auto& impl) { return GetHashValueImpl(impl); }, \
                      union_obj.variant());                                    \
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

#define ADT_TODO() PADDLE_THROW(::common::errors::Fatal("TODO"))

inline std::size_t hash_combine(std::size_t lhs, std::size_t rhs) {
  return lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
}

}  // namespace adt
}  // namespace cinn
