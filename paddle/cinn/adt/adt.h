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
  template <typename T>
  explicit Union(const T& x) : var_(x) {}

  template <typename... Fs>
  auto operator>>(match<Fs...> const& match) const {
    return var_ >> match;
  }

 private:
  std::variant<Ts...> var_;
};

template <typename... Ts>
class Tuple {
 public:
  template <typename... Args>
  explicit Tuple(Args&&... args)
      : tuple_(
            std::make_shared<std::tuple<Ts...>>(std::forward<Args>(args)...)) {}

 protected:
  std::shared_ptr<std::tuple<Ts...>> tuple_;
};

template <typename T>
using List = std::vector<T>;

template <typename T>
using Box = std::shared_ptr<T>;

template <typename T>
class Tagged {
 public:
  template <typename ValueT>
  explicit Tagged(ValueT&& value) : value_(value) {}

  const T& value() const { return value_; }

 private:
  T value_;
};

#define DEFINE_ADT_TAG(name)            \
  template <typename T>                 \
  class name final : public Tagged<T> { \
    using Tagged<T>::Tagged;            \
  };

DEFINE_ADT_TAG(In);
DEFINE_ADT_TAG(Out);
DEFINE_ADT_TAG(Optional);
DEFINE_ADT_TAG(tVar);
DEFINE_ADT_TAG(tSSAShadow);
DEFINE_ADT_TAG(tAnchor);
DEFINE_ADT_TAG(tScheduleIterVar);
DEFINE_ADT_TAG(tAssertMsg);
DEFINE_ADT_TAG(tIndexVar);
DEFINE_ADT_TAG(tTensorSize);
using Name = std::string;

}  // namespace adt
}  // namespace cinn
