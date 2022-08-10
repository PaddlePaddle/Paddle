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
#include <math.h>

namespace phi {
namespace funcs {

#define COMPARE_FUNCTOR(func_name, op)                           \
  template <typename InT, typename OutT = bool>                  \
  struct func_name {                                             \
    HOSTDEVICE OutT operator()(const InT a, const InT b) const { \
      return static_cast<OutT>(a op b);                          \
    }                                                            \
  };

COMPARE_FUNCTOR(LessThanFunctor, <)
COMPARE_FUNCTOR(LessEqualFunctor, <=)
COMPARE_FUNCTOR(GreaterThanFunctor, >)
COMPARE_FUNCTOR(GreaterEqualFunctor, >=)
#undef COMPARE_FUNCTOR

template <typename InT, typename OutT = bool>
struct EqualFunctor {
  HOSTDEVICE OutT operator()(const InT a, const InT b) const {
    if (std::is_floating_point<InT>::value) {
      if (isinf(static_cast<float>(a)) || isinf(static_cast<float>(b)))
        return static_cast<OutT>(a == b);
      if (isnan(static_cast<float>(a)) || isnan(static_cast<float>(b)))
        return static_cast<OutT>(false);
      return static_cast<OutT>(fabs(static_cast<double>(a - b)) < 1e-8);
    } else {
      return static_cast<OutT>(a == b);
    }
  }
};

template <typename InT, typename OutT = bool>
struct NotEqualFunctor {
  HOSTDEVICE bool operator()(const InT a, const InT b) const {
    return !EqualFunctor<InT, OutT>()(a, b);
  }
};

}  // namespace funcs
}  // namespace phi
