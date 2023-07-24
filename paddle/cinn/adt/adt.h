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
using Tuple = std::tuple<Ts...>;

template <typename T>
using List = std::vector<T>;

template <typename T>
using Box = std::shared_ptr<T>;

}  // namespace adt
}  // namespace cinn
