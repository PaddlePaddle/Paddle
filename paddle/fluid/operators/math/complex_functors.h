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

#include <type_traits>

#include "paddle/fluid/platform/complex.h"
#include "paddle/pten/core/hostdevice.h"

namespace paddle {
namespace operators {
namespace math {

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
    select_t<cond<std::is_same<T, platform::complex<float>>::value, float>,
             cond<std::is_same<T, platform::complex<double>>::value, double>,
             T>;

template <typename T, typename RealT>
using Complex = typename std::enable_if<!std::is_same<T, RealT>::value>::type;

// There are no NoComplex cases now, implement later if needed
template <typename T, typename RealT>
using NoComplex = typename std::enable_if<std::is_same<T, RealT>::value>::type;

template <typename T>
using EnableComplex = typename std::enable_if<
    std::is_same<T, platform::complex<float>>::value ||
    std::is_same<T, platform::complex<double>>::value>::type;

template <typename T>
using DisableComplex = typename std::enable_if<
    !std::is_same<T, platform::complex<float>>::value &&
    !std::is_same<T, platform::complex<double>>::value>::type;

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
  AbsGradFunctor(const math::Real<T>* dout, const T* x, T* output,
                 int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == T(0)) {
      output_[idx] = T(0);
    } else {
      output_[idx] = T(dout_[idx]) * (x_[idx] / T(std::abs(x_[idx])));
    }
  }

  const math::Real<T>* dout_;
  const T* x_;
  T* output_;
  int64_t numel_;
};

template <>
struct AbsGradFunctor<paddle::platform::complex<float>> {
  AbsGradFunctor(const float* dout, const paddle::platform::complex<float>* x,
                 paddle::platform::complex<float>* output, int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == paddle::platform::complex<float>(0)) {
      output_[idx] = paddle::platform::complex<float>(0);
    } else {
      output_[idx] = paddle::platform::complex<float>(dout_[idx]) *
                     (x_[idx] / paddle::platform::complex<float>(abs(x_[idx])));
    }
  }

  const float* dout_;
  const paddle::platform::complex<float>* x_;
  paddle::platform::complex<float>* output_;
  int64_t numel_;
};

template <>
struct AbsGradFunctor<paddle::platform::complex<double>> {
  AbsGradFunctor(const double* dout, const paddle::platform::complex<double>* x,
                 paddle::platform::complex<double>* output, int64_t numel)
      : dout_(dout), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == paddle::platform::complex<double>(0)) {
      output_[idx] = paddle::platform::complex<double>(0);
    } else {
      output_[idx] =
          paddle::platform::complex<double>(dout_[idx]) *
          (x_[idx] / paddle::platform::complex<double>(abs(x_[idx])));
    }
  }

  const double* dout_;
  const paddle::platform::complex<double>* x_;
  paddle::platform::complex<double>* output_;
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
struct AbsGradGradFunctor<paddle::platform::complex<double>> {
  AbsGradGradFunctor(const paddle::platform::complex<double>* ddx,
                     const paddle::platform::complex<double>* x,
                     paddle::platform::complex<double>* output, int64_t numel)
      : ddx_(ddx), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == paddle::platform::complex<double>(0)) {
      output_[idx] = paddle::platform::complex<double>(0);
    } else {
      output_[idx] = paddle::platform::complex<double>(ddx_[idx]) * x_[idx] /
                     paddle::platform::complex<double>(abs(x_[idx]));
    }
  }

  const paddle::platform::complex<double>* ddx_;
  const paddle::platform::complex<double>* x_;
  paddle::platform::complex<double>* output_;
  int64_t numel_;
};

template <>
struct AbsGradGradFunctor<paddle::platform::complex<float>> {
  AbsGradGradFunctor(const paddle::platform::complex<float>* ddx,
                     const paddle::platform::complex<float>* x,
                     paddle::platform::complex<float>* output, int64_t numel)
      : ddx_(ddx), x_(x), output_(output), numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    if (x_[idx] == paddle::platform::complex<float>(0)) {
      output_[idx] = paddle::platform::complex<float>(0);
    } else {
      output_[idx] = paddle::platform::complex<float>(ddx_[idx]) * x_[idx] /
                     paddle::platform::complex<float>(abs(x_[idx]));
    }
  }

  const paddle::platform::complex<float>* ddx_;
  const paddle::platform::complex<float>* x_;
  paddle::platform::complex<float>* output_;
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
  RealImagToComplexFunctor(const Real<T>* input_real, const Real<T>* input_imag,
                           T* output, int64_t numel)
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

}  // namespace math
}  // namespace operators
}  // namespace paddle
