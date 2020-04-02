/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/operators/math.h"

namespace paddle {
namespace operators {
namespace math {

// MulFunctor
template <typename T>
struct MulFunctor {
  // out = x * y;
  inline HOSTDEVICE T operator()(T x, T y) { return x * y; }
};

template <typename T>
struct MulGradFunctor {
  inline HOSTDEVICE T Dx(T x, T y) { return y; }
  inline HOSTDEVICE T Dy(T x, T y) { return x; }
};

// AddFunctor
template <typename T>
struct AddFunctor {
  // out = x + y;
  inline HOSTDEVICE T operator()(T x, T y) { return x + y; }
};

template <typename T>
struct AddGradFunctor {
  inline HOSTDEVICE T Dx(T x, T y) { return static_cast<T>(1.); }
  inline HOSTDEVICE T Dy(T x, T y) { return static_cast<T>(1.); }
};

template <typename T>
struct ScaleFunctor {
  explicit ScaleFunctor(const T coeff) : coeff_(coeff) {}

  inline HOSTDEVICE T operator()(T ele) { return ele * coeff_; }

 private:
  T coeff_;
};

template <typename T>
struct ScaleGradFunctor {
  explicit ScaleGradFunctor(T coeff) : coeff_(coeff) {}

  inline HOSTDEVICE T UseX(T x) { return coeff_; }
  inline HOSTDEVICE T UseOut(T out) { return coeff_; }
  inline HOSTDEVICE T UseXAndOut(T x, T out) { return coeff_; }

 private:
  T coeff_;
};

template <typename T>
struct ReluFunctor {
  inline HOSTDEVICE T operator()(T x) {
    return x * (x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0));
  }
};

template <typename T>
struct ReluGradFunctor {
  inline HOSTDEVICE T UseX(T x) {
    return x > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);
  }
  inline HOSTDEVICE T UseOut(T out) {
    return out > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);
  }
  inline HOSTDEVICE T UseXAndOut(T x, T out) {
    return out > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(0);
  }
};

template <typename T>
struct TanhFunctor {
  const T kMin = static_cast<T>(-40);
  const T kMax = static_cast<T>(13);
  inline HOSTDEVICE T operator()(T x) {
    // y = 2 / (1 + e^-2x) - 1
    T t0 = static_cast<T>(2) * x;
    T t1 = (t0 < kMin) ? kMin : ((t0 > kMax) ? kMax : t0);
    return static_cast<T>(2) / (static_cast<T>(1) + real_exp(-t1)) -
           static_cast<T>(1);
  }
};

template <typename T>
struct TanhGradFunctor {
  inline HOSTDEVICE T UseX(T x) { return static_cast<T>(1) - x * x; }
  inline HOSTDEVICE T UseOut(T out) { return static_cast<T>(1) - out * out; }
  inline HOSTDEVICE T UseXAndOut(T x, T out) {
    return static_cast<T>(1) - out * out;
  }
};

template <typename T>
struct SigmoidFunctor {
  const T kMin = static_cast<T>(-40);
  const T kMax = static_cast<T>(13);
  inline HOSTDEVICE T operator()(T x) {
    // y = 1 / (1 + e^-x)
    T tmp = (x < kMin) ? kMin : ((x > kMax) ? kMax : x);
    return static_cast<T>(1) / (static_cast<T>(1) + real_exp(-tmp));
  }
};

template <typename T>
struct SigmoidGradFunctor {
  inline HOSTDEVICE T UseX(T x) { return x * (static_cast<T>(1) - x); }
  inline HOSTDEVICE T UseOut(T out) { return out * (static_cast<T>(1) - out); }
  inline HOSTDEVICE T UseXAndOut(T x, T out) {
    return out * (static_cast<T>(1) - out);
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
