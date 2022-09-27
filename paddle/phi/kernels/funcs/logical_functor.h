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

#define LOGICAL_BINARY_FUNCTOR(func_name, op)                \
  template <typename T>                                      \
  struct func_name {                                         \
    using ELEMENT_TYPE = T;                                  \
    HOSTDEVICE bool operator()(const T a, const T b) const { \
      return static_cast<bool>(a) op static_cast<bool>(b);   \
    }                                                        \
  };

LOGICAL_BINARY_FUNCTOR(LogicalOrFunctor, ||)
LOGICAL_BINARY_FUNCTOR(LogicalAndFunctor, &&)
LOGICAL_BINARY_FUNCTOR(LogicalXorFunctor, ^)
#undef LOGICAL_BINARY_FUNCTOR

template <typename T>
struct LogicalNotFunctor {
  using ELEMENT_TYPE = T;
  HOSTDEVICE bool operator()(const T a) const { return !a; }
};

}  // namespace funcs
}  // namespace phi
