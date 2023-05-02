// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace phi {
namespace funcs {

#define BITWISE_BINARY_FUNCTOR(func, expr, bool_expr)                        \
  template <typename T>                                                      \
  struct Bitwise##func##Functor {                                            \
    HOSTDEVICE T operator()(const T a, const T b) const { return a expr b; } \
  };                                                                         \
                                                                             \
  template <>                                                                \
  struct Bitwise##func##Functor<bool> {                                      \
    HOSTDEVICE bool operator()(const bool a, const bool b) const {           \
      return a bool_expr b;                                                  \
    }                                                                        \
  };

BITWISE_BINARY_FUNCTOR(And, &, &&)
BITWISE_BINARY_FUNCTOR(Or, |, ||)
BITWISE_BINARY_FUNCTOR(Xor, ^, !=)
#undef BITWISE_BINARY_FUNCTOR

template <typename T>
struct BitwiseNotFunctor {
  using ELEM_TYPE = T;
  HOSTDEVICE T operator()(const T a) const { return ~a; }
};

template <>
struct BitwiseNotFunctor<bool> {
  using ELEM_TYPE = bool;
  HOSTDEVICE bool operator()(const bool a) const { return !a; }
};

}  // namespace funcs
}  // namespace phi
