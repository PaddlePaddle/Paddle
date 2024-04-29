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

template <typename T>
struct BitwiseLeftShiftArithmeticFunctor {
  HOSTDEVICE T operator()(const T a, const T b) const {
    if (b >= static_cast<T>(sizeof(T) * 8)) return static_cast<T>(0);
    if (b < static_cast<T>(0)) return static_cast<T>(0);
    return a << b;
  }
};

template <typename T>
struct InverseBitwiseLeftShiftArithmeticFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    if (a >= static_cast<T>(sizeof(T) * 8)) return static_cast<T>(0);
    if (a < static_cast<T>(0)) return static_cast<T>(0);
    return b << a;
  }
};

template <typename T>
struct BitwiseLeftShiftLogicFunctor {
  HOSTDEVICE T operator()(const T a, const T b) const {
    if (b < static_cast<T>(0) || b >= static_cast<T>(sizeof(T) * 8))
      return static_cast<T>(0);
    return a << b;
  }
};

template <typename T>
struct InverseBitwiseLeftShiftLogicFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    if (a < static_cast<T>(0) || a >= static_cast<T>(sizeof(T) * 8))
      return static_cast<T>(0);
    return b << a;
  }
};

template <typename T>
struct BitwiseRightShiftArithmeticFunctor {
  HOSTDEVICE T operator()(const T a, const T b) const {
    if (b < static_cast<T>(0) || b >= static_cast<T>(sizeof(T) * 8))
      return static_cast<T>(-(a >> (sizeof(T) * 8 - 1) & 1));
    return a >> b;
  }
};

template <typename T>
struct InverseBitwiseRightShiftArithmeticFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    if (a < static_cast<T>(0) || a >= static_cast<T>(sizeof(T) * 8))
      return static_cast<T>(-(b >> (sizeof(T) * 8 - 1) & 1));
    return b >> a;
  }
};

template <>
struct BitwiseRightShiftArithmeticFunctor<uint8_t> {
  HOSTDEVICE uint8_t operator()(const uint8_t a, const uint8_t b) const {
    if (b >= static_cast<uint8_t>(sizeof(uint8_t) * 8))
      return static_cast<uint8_t>(0);
    return a >> b;
  }
};

template <>
struct InverseBitwiseRightShiftArithmeticFunctor<uint8_t> {
  inline HOSTDEVICE uint8_t operator()(const uint8_t a, const uint8_t b) const {
    if (a >= static_cast<uint8_t>(sizeof(uint8_t) * 8))
      return static_cast<uint8_t>(0);
    return b >> a;
  }
};

template <typename T>
struct BitwiseRightShiftLogicFunctor {
  HOSTDEVICE T operator()(const T a, const T b) const {
    if (b >= static_cast<T>(sizeof(T) * 8) || b < static_cast<T>(0))
      return static_cast<T>(0);
    return a >> b;
  }
};

template <typename T>
struct InverseBitwiseRightShiftLogicFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    if (a >= static_cast<T>(sizeof(T) * 8) || a < static_cast<T>(0))
      return static_cast<T>(0);
    return b >> a;
  }
};

template <typename T>
HOSTDEVICE T logic_shift_func(const T a, const T b) {
  if (b < static_cast<T>(0) || b >= static_cast<T>(sizeof(T) * 8))
    return static_cast<T>(0);
  T t = static_cast<T>(sizeof(T) * 8 - 1);
  T mask = (((a >> t) << t) >> b) << 1;
  return (a >> b) ^ mask;
}

// signed int8
template <>
struct BitwiseRightShiftLogicFunctor<int8_t> {
  HOSTDEVICE int8_t operator()(const int8_t a, const int8_t b) const {
    return logic_shift_func<int8_t>(a, b);
  }
};

template <>
struct InverseBitwiseRightShiftLogicFunctor<int8_t> {
  inline HOSTDEVICE int8_t operator()(const int8_t a, const int8_t b) const {
    return logic_shift_func<int8_t>(b, a);
  }
};

// signed int16
template <>
struct BitwiseRightShiftLogicFunctor<int16_t> {
  HOSTDEVICE int16_t operator()(const int16_t a, const int16_t b) const {
    return logic_shift_func<int16_t>(a, b);
  }
};

template <>
struct InverseBitwiseRightShiftLogicFunctor<int16_t> {
  inline HOSTDEVICE int16_t operator()(const int16_t a, const int16_t b) const {
    return logic_shift_func<int16_t>(b, a);
  }
};

// signed int32
template <>
struct BitwiseRightShiftLogicFunctor<int> {
  HOSTDEVICE int operator()(const int a, const int b) const {
    return logic_shift_func<int32_t>(a, b);
  }
};

template <>
struct InverseBitwiseRightShiftLogicFunctor<int> {
  inline HOSTDEVICE int operator()(const int a, const int b) const {
    return logic_shift_func<int32_t>(b, a);
  }
};

// signed int64
template <>
struct BitwiseRightShiftLogicFunctor<int64_t> {
  HOSTDEVICE int64_t operator()(const int64_t a, const int64_t b) const {
    return logic_shift_func<int64_t>(a, b);
  }
};

template <>
struct InverseBitwiseRightShiftLogicFunctor<int64_t> {
  inline HOSTDEVICE int64_t operator()(const int64_t a, const int64_t b) const {
    return logic_shift_func<int64_t>(b, a);
  }
};

}  // namespace funcs
}  // namespace phi
