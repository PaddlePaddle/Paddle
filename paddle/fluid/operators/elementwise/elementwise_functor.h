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

#include "paddle/fluid/framework/array.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/pten/kernels/funcs/elementwise_functor.h"

namespace paddle {
namespace operators {

// Define the binary functors used in elementwise ops.

// Add
template <typename T>
using AddFunctor = pten::funcs::AddFunctor<T>;

template <typename T>
using InverseAddFunctor = pten::funcs::InverseAddFunctor<T>;

// Subtract
template <typename T>
using SubFunctor = pten::funcs::SubtractFunctor<T>;

template <typename T>
using InverseSubFunctor = pten::funcs::InverseSubtractFunctor<T>;

// Multiply
template <typename T>
using MulFunctor = pten::funcs::MultiplyFunctor<T>;

template <typename T>
using InverseMulFunctor = pten::funcs::InverseMultiplyFunctor<T>;

// Divide
template <typename T>
using DivFunctor = pten::funcs::DivideFunctor<T>;

template <typename T>
using InverseDivFunctor = pten::funcs::InverseDivideFunctor<T>;

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

template <typename T>
using Complex = paddle::platform::complex<T>;

template <typename InT, typename OutT>
struct DivGradXYFunctor {
  inline HOSTDEVICE paddle::framework::Array<OutT, 2> operator()(const InT a,
                                                                 const InT b,
                                                                 const InT c) {
    // dx = dout / y
    // dy = - dout * out / y
    paddle::framework::Array<OutT, 2> outs;
    outs[0] = a / c;
    outs[1] = -a * b / c;
    return outs;
  }
};

template <typename InT, typename OutT>
struct DivGradXYFunctor<Complex<InT>, Complex<OutT>> {
  inline HOSTDEVICE paddle::framework::Array<Complex<OutT>, 2> operator()(
      const Complex<InT> a, const Complex<InT> b, const Complex<InT> c) {
    paddle::framework::Array<Complex<OutT>, 2> outs;
    Complex<InT> c_conj(c.real, -c.imag);
    Complex<InT> out_div_c_conj((b / c).real, -(b / c).imag);
    outs[0] = a / c_conj;
    outs[1] = -a * out_div_c_conj;
    return outs;
  }
};

// Float div grad
template <typename T>
struct DivGradXFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a / b; }
};

// Complex div grad
template <typename T>
struct DivGradXFunctor<Complex<T>> {
  inline HOSTDEVICE Complex<T> operator()(const Complex<T>& a,
                                          const Complex<T>& b) const {
    Complex<T> b_conj(b.real, -b.imag);
    return a / b_conj;
  }
};

// Float mul and div
template <typename T>
struct DivGradYFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b, const T& c) const {
    return -a * b / c;
  }
};

// Complex mul and div
template <typename T>
struct DivGradYFunctor<Complex<T>> {
  inline HOSTDEVICE Complex<T> operator()(const Complex<T>& a,
                                          const Complex<T>& b,
                                          const Complex<T>& c) const {
    Complex<T> out_div_c_conj((b / c).real, -(b / c).imag);
    return -a * out_div_c_conj;
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

template <typename T>
struct MulGradFunctor {
  inline HOSTDEVICE T operator()(const T& a, const T& b) const { return a * b; }
};
template <typename T>
struct MulGradFunctor<Complex<T>> {
  inline HOSTDEVICE Complex<T> operator()(const Complex<T>& a,
                                          const Complex<T>& b) const {
    Complex<T> b_conj(b.real, -b.imag);
    return a * b_conj;
  }
};

template <typename InT, typename OutT>
struct MulGradXYFunctor {
  inline HOSTDEVICE paddle::framework::Array<OutT, 2> operator()(const InT& a,
                                                                 const InT& b,
                                                                 const InT& c) {
    paddle::framework::Array<OutT, 2> outs;
    // dx = dout * y
    outs[0] = a * b;
    // dy = dout * x
    outs[1] = a * c;
    return outs;
  }
};

template <typename InT, typename OutT>
struct MulGradXYFunctor<Complex<InT>, Complex<OutT>> {
  inline HOSTDEVICE paddle::framework::Array<Complex<OutT>, 2> operator()(
      const Complex<InT>& a, const Complex<InT>& b, const Complex<InT>& c) {
    paddle::framework::Array<Complex<OutT>, 2> outs;
    // dx = dout * y
    Complex<InT> b_conj(b.real, -b.imag);
    outs[0] = a * b_conj;
    // dy = dout * x
    Complex<InT> c_conj(c.real, -c.imag);
    outs[1] = a * c_conj;
    return outs;
  }
};

}  // namespace operators
}  // namespace paddle
