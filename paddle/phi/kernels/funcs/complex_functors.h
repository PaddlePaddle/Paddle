/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <type_traits>

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/hostdevice.h"

namespace pten {
namespace funcs {

template <bool B, typename T>
struct cond {
  static constexpr bool value = B;
  using type = T;
};

template <bool B, typename TrueF, typename FalseF>
struct eval_if {
  using type = typename TrueF::type;
};

template <typename TrueF, typename FalseF>
struct eval_if<false, TrueF, FalseF> {
  using type = typename FalseF::type;
};

template <bool B, typename T, typename F>
using eval_if_t = typename eval_if<B, T, F>::type;

template <typename Head, typename... Tail>
struct select {
  using type = eval_if_t<Head::value, Head, select<Tail...>>;
};

template <typename T>
struct select<T> {
  using type = T;
};

template <bool B, typename T>
struct select<cond<B, T>> {
  // last one had better be true!
  static_assert(B, "No match select type!");
  using type = T;
};

template <typename Head, typename... Tail>
using select_t = typename select<Head, Tail...>::type;

template <typename T>
using Real =
    select_t<cond<std::is_same<T, pten::dtype::complex<float>>::value, float>,
             cond<std::is_same<T, pten::dtype::complex<double>>::value, double>,
             T>;

template <typename T, typename RealT>
using Complex = typename std::enable_if<!std::is_same<T, RealT>::value>::type;

// There are no NoComplex cases now, implement later if needed
template <typename T, typename RealT>
using NoComplex = typename std::enable_if<std::is_same<T, RealT>::value>::type;

template <typename T>
using EnableComplex = typename std::enable_if<
    std::is_same<T, pten::dtype::complex<float>>::value ||
    std::is_same<T, pten::dtype::complex<double>>::value>::type;

template <typename T>
using DisableComplex = typename std::enable_if<
    !std::is_same<T, pten::dtype::complex<float>>::value &&
    !std::is_same<T, pten::dtype::complex<double>>::value>::type;

template <typename T, typename Enable = void>
struct RealFunctor;

template <typename T>
struct RealFunctor<T, Complex<T, Real<T>>> {
 public:
  RealFunctor(const T* input, Real<T>* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = input_[idx].real;
  }

 private:
  const T* input_;
  Real<T>* output_;
  int64_t numel_;
};

template <typename T, typename Enable = void>
struct ImagFunctor;

template <typename T>
struct ImagFunctor<T, Complex<T, Real<T>>> {
  ImagFunctor(const T* input, Real<T>* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = input_[idx].imag;
  }

  const T* input_;
  Real<T>* output_;
  int64_t numel_;
};

template <typename T, typename Enable = void>
struct AbsFunctor;

template <typename T>
struct AbsFunctor<T, Complex<T, Real<T>>> {
  AbsFunctor(const T* input, Real<T>* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = abs(input_[idx]);
  }

  const T* input_;
  Real<T>* output_;
  int64_t numel_;
};

template <typename T>
struct AbsFunctor<T, NoComplex<T, Real<T>>> {
  AbsFunctor(const T* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = std::abs(input_[idx]);
  }

  const T* input_;
  T* output_;
  int64_t numel_;
};

template <typename T>
struct AbsGradFunctor {
  AbsGradFunctor(const Real<T>* dout, const T* x, T* output, int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == T(0)) {
      output_[idx] = T(0);
    } else {
      output_[idx] = T(dout_[idx]) * (x_[idx] / T(std::abs(x_[idx])));
    }
  }

  const Real<T>* dout_;
  const T* x_;
  T* output_;
  int64_t numel_;
};

template <>
struct AbsGradFunctor<pten::dtype::complex<float>> {
  AbsGradFunctor(const float* dout,
                 const pten::dtype::complex<float>* x,
                 pten::dtype::complex<float>* output,
                 int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == pten::dtype::complex<float>(0)) {
      output_[idx] = pten::dtype::complex<float>(0);
    } else {
      output_[idx] = pten::dtype::complex<float>(dout_[idx]) *
                     (x_[idx] / pten::dtype::complex<float>(abs(x_[idx])));
    }
  }

  const float* dout_;
  const pten::dtype::complex<float>* x_;
  pten::dtype::complex<float>* output_;
  int64_t numel_;
};

template <>
struct AbsGradFunctor<pten::dtype::complex<double>> {
  AbsGradFunctor(const double* dout,
                 const pten::dtype::complex<double>* x,
                 pten::dtype::complex<double>* output,
                 int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == pten::dtype::complex<double>(0)) {
      output_[idx] = pten::dtype::complex<double>(0);
    } else {
      output_[idx] = pten::dtype::complex<double>(dout_[idx]) *
                     (x_[idx] / pten::dtype::complex<double>(abs(x_[idx])));
    }
  }

  const double* dout_;
  const pten::dtype::complex<double>* x_;
  pten::dtype::complex<double>* output_;
  int64_t numel_;
};

template <typename T>
struct AbsGradGradFunctor {
  AbsGradGradFunctor(const T* ddx, const T* x, T* output, int64_t numel)
      : ddx_(ddx), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == T(0)) {
      output_[idx] = T(0);
    } else {
      output_[idx] = T(ddx_[idx]) * x_[idx] / T(std::abs(x_[idx]));
    }
  }

  const T* ddx_;
  const T* x_;
  T* output_;
  int64_t numel_;
};

template <>
struct AbsGradGradFunctor<pten::dtype::complex<double>> {
  AbsGradGradFunctor(const pten::dtype::complex<double>* ddx,
                     const pten::dtype::complex<double>* x,
                     pten::dtype::complex<double>* output,
                     int64_t numel)
      : ddx_(ddx), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == pten::dtype::complex<double>(0)) {
      output_[idx] = pten::dtype::complex<double>(0);
    } else {
      output_[idx] = pten::dtype::complex<double>(ddx_[idx]) * x_[idx] /
                     pten::dtype::complex<double>(abs(x_[idx]));
    }
  }

  const pten::dtype::complex<double>* ddx_;
  const pten::dtype::complex<double>* x_;
  pten::dtype::complex<double>* output_;
  int64_t numel_;
};

template <>
struct AbsGradGradFunctor<pten::dtype::complex<float>> {
  AbsGradGradFunctor(const pten::dtype::complex<float>* ddx,
                     const pten::dtype::complex<float>* x,
                     pten::dtype::complex<float>* output,
                     int64_t numel)
      : ddx_(ddx), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == pten::dtype::complex<float>(0)) {
      output_[idx] = pten::dtype::complex<float>(0);
    } else {
      output_[idx] = pten::dtype::complex<float>(ddx_[idx]) * x_[idx] /
                     pten::dtype::complex<float>(abs(x_[idx]));
    }
  }

  const pten::dtype::complex<float>* ddx_;
  const pten::dtype::complex<float>* x_;
  pten::dtype::complex<float>* output_;
  int64_t numel_;
};
template <typename T, typename Enable = void>
struct RealToComplexFunctor;

template <typename T>
struct RealToComplexFunctor<T, Complex<T, Real<T>>> {
  RealToComplexFunctor(const Real<T>* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx].real = input_[idx];
    output_[idx].imag = 0;
  }

  const Real<T>* input_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Enable = void>
struct ImagToComplexFunctor;

template <typename T>
struct ImagToComplexFunctor<T, Complex<T, Real<T>>> {
  ImagToComplexFunctor(const Real<T>* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx].real = 0;
    output_[idx].imag = input_[idx];
  }

  const Real<T>* input_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Enable = void>
struct RealImagToComplexFunctor;

template <typename T>
struct RealImagToComplexFunctor<T, Complex<T, Real<T>>> {
  RealImagToComplexFunctor(const Real<T>* input_real,
                           const Real<T>* input_imag,
                           T* output,
                           int64_t numel)
      : input_real_(input_real),
        input_imag_(input_imag),
        output_(output),
        numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx].real = input_real_[idx];
    output_[idx].imag = input_imag_[idx];
  }

  const Real<T>* input_real_;
  const Real<T>* input_imag_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Enable = void>
struct ConjFunctor;

template <typename T>
struct ConjFunctor<T, EnableComplex<T>> {
  ConjFunctor(const T* input, int64_t numel, T* output)
      : input_(input), numel_(numel), output_(output) {}

  HOSTDEVICE void operator()(size_t idx) const {
    output_[idx] = T(input_[idx].real, -input_[idx].imag);
  }
  const T* input_;
  int64_t numel_;
  T* output_;
};

template <typename T>
struct ConjFunctor<T, DisableComplex<T>> {
  ConjFunctor(const T* input, int64_t numel, T* output)
      : input_(input), numel_(numel), output_(output) {}

  HOSTDEVICE void operator()(size_t idx) const { output_[idx] = input_[idx]; }
  const T* input_;
  int64_t numel_;
  T* output_;
};

template <typename T, typename Enable = void>
struct AngleFunctor;

// angel function for complex
template <typename T>
struct AngleFunctor<T, pten::funcs::Complex<T, pten::funcs::Real<T>>> {
  AngleFunctor(const T* input, pten::funcs::Real<T>* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = arg(input_[idx]);
  }

  const T* input_;
  pten::funcs::Real<T>* output_;
  int64_t numel_;
};

// angel function for real
template <typename T>
struct AngleFunctor<T, pten::funcs::NoComplex<T, pten::funcs::Real<T>>> {
  AngleFunctor(const T* input, T* output, int64_t numel)
      : input_(input), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = input_[idx] < static_cast<T>(0) ? M_PI : 0;
  }

  const T* input_;
  T* output_;
  int64_t numel_;
};

template <typename T, typename Enable = void>
struct AngleGradFunctor;

// angle grad for complex
template <typename T>
struct AngleGradFunctor<T, pten::funcs::Complex<T, pten::funcs::Real<T>>> {
  AngleGradFunctor(const pten::funcs::Real<T>* dout,
                   const T* x,
                   T* dx,
                   int64_t numel)
      : dout_(dout), x_(x), dx_(dx), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == T(0)) {
      dx_[idx] = T(0);
    } else {
      const pten::funcs::Real<T> r_square =
          x_[idx].real * x_[idx].real + x_[idx].imag * x_[idx].imag;
      dx_[idx] = T(-dout_[idx] * x_[idx].imag / r_square,
                   dout_[idx] * x_[idx].real / r_square);
    }
  }

  const pten::funcs::Real<T>* dout_;
  const T* x_;
  T* dx_;
  int64_t numel_;
};

// angle grad for real
template <typename T>
struct AngleGradFunctor<T, pten::funcs::NoComplex<T, pten::funcs::Real<T>>> {
  AngleGradFunctor(const pten::funcs::Real<T>* dout,
                   const T* x,
                   T* dx,
                   int64_t numel)
      : dout_(dout), x_(x), dx_(dx), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const { dx_[idx] = 0; }

  const pten::funcs::Real<T>* dout_;
  const T* x_;
  T* dx_;
  int64_t numel_;
};

}  // namespace funcs
}  // namespace pten
