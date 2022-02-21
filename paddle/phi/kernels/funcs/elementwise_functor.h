/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/pten/common/float16.h"
#include "paddle/pten/core/enforce.h"
#include "paddle/pten/core/hostdevice.h"

namespace pten {
namespace funcs {

// Define the binary functors used in elementwise ops.
// Note: InverseXxxFunctor is needed when calling ElementwiseComputeEx on CPU.

// Add
template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return a + b; }
};
template <typename T>
struct InverseAddFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return b + a; }
};

// Subtract
template <typename T>
struct SubtractFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return a - b; }
};
template <typename T>
struct InverseSubtractFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return b - a; }
};

// Multiply
template <typename T>
struct MultiplyFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return a * b; }
};
template <>
struct MultiplyFunctor<bool> {
  inline HOSTDEVICE bool operator()(const bool a, const bool b) const {
    return a && b;
  }
};
template <typename T>
struct InverseMultiplyFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return b * a; }
};
template <>
struct InverseMultiplyFunctor<bool> {
  inline HOSTDEVICE bool operator()(const bool a, const bool b) const {
    return b && a;
  }
};

// Divide
#define DIV_ERROR_INFO                                             \
  "InvalidArgumentError: Integer division by zero encountered in " \
  "(floor) divide. Please check the input value."

template <typename T, typename Enable = void>
struct DivideFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return a / b; }
};

template <typename T>
struct DivideFunctor<
    T,
    typename std::enable_if<std::is_integral<T>::value>::type> {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    // For int32/int64, need to check whether the divison is zero.
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
    return a / b;
  }
};

template <typename T, typename Enable = void>
struct InverseDivideFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const { return b / a; }
};

}  // namespace funcs
}  // namespace pten
