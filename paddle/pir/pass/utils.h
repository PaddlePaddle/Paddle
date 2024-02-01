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

#include <ostream>
#include <string>
#include <type_traits>

namespace pir {
namespace detail {

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;
template <class, template <class...> class Op, class... Args>
struct detector {
  using value_t = std::false_type;
};

template <template <class...> class Op, class... Args>
struct detector<void_t<Op<Args...>>, Op, Args...> {
  using value_t = std::true_type;
};

template <template <class...> class Op, class... Args>
using is_detected = typename detector<void, Op, Args...>::value_t;

// Print content as follows.
// ===-------------------------------------------------------------------------===
//                                     header
// ===-------------------------------------------------------------------------===
void PrintHeader(const std::string &header, std::ostream &os);

}  // namespace detail
}  // namespace pir
