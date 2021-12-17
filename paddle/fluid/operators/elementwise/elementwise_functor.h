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

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

// Define the binary functors used in elementwise ops.

// Add
template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a + b; }
};
template <typename T>
struct InverseAddFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b + a; }
};

// Subtract
template <typename T>
struct SubFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a - b; }
};
template <typename T>
struct InverseSubFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b - a; }
};

// Multiply
template <typename T>
struct MulFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a * b; }
};
template <typename T>
struct InverseMulFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b * a; }
};

// Divide
#define DIV_ERROR_INFO                                             \
  "InvalidArgumentError: Integer division by zero encountered in " \
  "(floor) divide. Please check the input value."

template <typename T, typename Enable = void>
struct DivFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a / b; }
};

template <typename T>
struct DivFunctor<T,
                  typename std::enable_if<std::is_integral<T>::value>::type> {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    // For int32/int64, need to check whether the divison is zero.
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
    return a / b;
  }
};

template <typename T, typename Enable = void>
struct InverseDivFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return b / a; }
};

// Floor Divide
template <typename T>
struct FloorDivFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    PADDLE_ENFORCE(b != 0, DIV_ERROR_INFO);
    return static_cast<T>(std::trunc(a / b));
  }
};

template <typename T>
struct InverseFloorDivFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    PADDLE_ENFORCE(a != 0, DIV_ERROR_INFO);
    return static_cast<T>(std::trunc(b / a));
  }
};

#undef DIV_ERROR_INFO

// Maximum
template <typename T>
struct MaxFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    return a > b ? a : b;
  }
};

// Minmum
template <typename T>
struct MinFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    return a < b ? a : b;
  }
};

// Fmax
template <typename T>
struct FMaxFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    return std::fmax(a, b);
  }
};

template <>
struct FMaxFunctor<paddle::platform::float16> {
  inline HOSTDEVICE paddle::platform::float16 operator()(
      const paddle::platform::float16& a,
      const paddle::platform::float16& b) const {
    float float_a = static_cast<float>(a);
    float float_b = static_cast<float>(b);
    auto result = std::fmax(float_a, float_b);
    return static_cast<paddle::platform::float16>(result);
  }
};

// Fmin
template <typename T>
struct FMinFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const {
    return std::fmin(a, b);
  }
};

template <>
struct FMinFunctor<paddle::platform::float16> {
  inline HOSTDEVICE paddle::platform::float16 operator()(
      const paddle::platform::float16& a,
      const paddle::platform::float16& b) const {
    float float_a = static_cast<float>(a);
    float float_b = static_cast<float>(b);
    auto result = std::fmin(float_a, float_b);
    return static_cast<paddle::platform::float16>(result);
  }
};

}  // namespace operators
}  // namespace paddle
