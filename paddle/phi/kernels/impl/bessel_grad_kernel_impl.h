/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/impl/bessel_kernel_impl.h"

namespace phi {

template <typename T>
struct I0GradFunctor {
  I0GradFunctor(const T* x, const T* out_grad, T* x_grad, int64_t numel)
      : inp_x_(x),
        inp_out_grad_(out_grad),
        output_x_grad_(x_grad),
        numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    const MT mp_x = static_cast<MT>(inp_x_[idx]);
    const MT mp_out_grad = static_cast<MT>(inp_out_grad_[idx]);

    MT x = std::abs(mp_x);
    if (x <= T{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI1e_A<MT>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      MT y = (x / MT{2.0}) - MT{2.0};

      const MT i1_out = std::exp(x) * x * Chbevl<MT>(y, A, len);
      const MT i1_data = (mp_x < T{0.0}) ? -i1_out : i1_out;
      output_x_grad_[idx] = static_cast<T>(i1_data * mp_out_grad);
    } else {
      auto coeff_pair_B = ChebyshevCoefficientsI1e_B<MT>();
      auto B = std::get<0>(coeff_pair_B);
      auto len = std::get<1>(coeff_pair_B);
      MT y = (MT{32.0} / x) - MT{2.0};

      const MT i1_out = (std::exp(x) * Chbevl<MT>(y, B, len)) / std::sqrt(x);
      const MT i1_data = (mp_x < MT{0.0}) ? -i1_out : i1_out;
      output_x_grad_[idx] = static_cast<T>(i1_data * mp_out_grad);
    }
  }

 private:
  const T* inp_x_;
  const T* inp_out_grad_;
  T* output_x_grad_;
  int64_t numel_;
};

template <typename T>
struct I0eGradFunctor {
  I0eGradFunctor(
      const T* x, const T* out, const T* out_grad, T* x_grad, int64_t numel)
      : inp_x_(x),
        inp_out_(out),
        inp_out_grad_(out_grad),
        output_x_grad_(x_grad),
        numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    T x = std::abs(inp_x_[idx]);
    if (x <= T{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI1e_A<T>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      T y = (x / T{2.0}) - T{2.0};

      const T out = Chbevl<T>(y, A, len) * x;
      const T i1e_out = (inp_x_[idx] < T{0.0}) ? -out : out;
      output_x_grad_[idx] =
          (i1e_out - std::copysign(T{1.0}, inp_x_[idx]) * inp_out_[idx]) *
          inp_out_grad_[idx];
    } else {
      auto coeff_pair_B = ChebyshevCoefficientsI1e_B<T>();
      auto B = std::get<0>(coeff_pair_B);
      auto len = std::get<1>(coeff_pair_B);
      T y = (T{32.0} / x) - T{2.0};

      const T out = Chbevl<T>(y, B, len) / std::sqrt(x);
      const T i1e_out = (inp_x_[idx] < T{0.0}) ? -out : out;
      output_x_grad_[idx] =
          (i1e_out - std::copysign(T{1.0}, inp_x_[idx]) * inp_out_[idx]) *
          inp_out_grad_[idx];
    }
  }

 private:
  const T* inp_x_;
  const T* inp_out_;
  const T* inp_out_grad_;
  T* output_x_grad_;
  int64_t numel_;
};

template <typename T>
struct I1GradFunctor {
  I1GradFunctor(
      const T* x, const T* out, const T* out_grad, T* x_grad, int64_t numel)
      : input_x_(x),
        input_out_(out),
        input_out_grad_(out_grad),
        output_x_grad_(x_grad),
        numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    T x = std::abs(input_x_[idx]);
    T x_ = input_x_[idx];
    T out_ = input_out_[idx];
    T out_grad_ = input_out_grad_[idx];
    if (x <= T{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI0e_A<T>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      T y = (x / T{2.0}) - T{2.0};
      T eps = std::numeric_limits<T>::epsilon();

      if (x <= eps) {
        output_x_grad_[idx] = static_cast<T>(T{0.5} * out_grad_);
      } else {
        output_x_grad_[idx] = static_cast<T>(
            (std::exp(x) * Chbevl<T>(y, A, len) - out_ / x_) * out_grad_);
      }
    } else {
      auto coeff_pair_B = ChebyshevCoefficientsI0e_B<T>();
      auto B = std::get<0>(coeff_pair_B);
      auto len = std::get<1>(coeff_pair_B);
      T y = (T{32.0} / x) - T{2.0};

      output_x_grad_[idx] = static_cast<T>(
          (std::exp(x) * Chbevl<T>(y, B, len) / std::sqrt(x) - out_ / x_) *
          out_grad_);
    }
  }

 private:
  const T* input_x_;
  const T* input_out_;
  const T* input_out_grad_;
  T* output_x_grad_;
  int64_t numel_;
};

template <typename T>
struct I1eGradFunctor {
  I1eGradFunctor(
      const T* x, const T* out, const T* out_grad, T* x_grad, int64_t numel)
      : input_x_(x),
        input_out_(out),
        input_out_grad_(out_grad),
        output_x_grad_(x_grad),
        numel_(numel) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    T x = std::abs(input_x_[idx]);
    T x_ = input_x_[idx];
    T out_ = input_out_[idx];
    T out_grad_ = input_out_grad_[idx];
    if (x <= T{8.0}) {
      auto coeff_pair_A = ChebyshevCoefficientsI0e_A<T>();
      auto A = std::get<0>(coeff_pair_A);
      auto len = std::get<1>(coeff_pair_A);
      T y = (x / T{2.0}) - T{2.0};
      T eps = std::numeric_limits<T>::epsilon();

      if (x <= eps) {
        output_x_grad_[idx] = static_cast<T>(T{0.5} * out_grad_);
      } else {
        output_x_grad_[idx] =
            static_cast<T>((Chbevl<T>(y, A, len) -
                            out_ * (std::copysign(T{1.0}, x_) + T{1.0} / x_)) *
                           out_grad_);
      }
    } else {
      auto coeff_pair_B = ChebyshevCoefficientsI0e_B<T>();
      auto B = std::get<0>(coeff_pair_B);
      auto len = std::get<1>(coeff_pair_B);
      T y = (T{32.0} / x) - T{2.0};

      output_x_grad_[idx] =
          static_cast<T>((Chbevl<T>(y, B, len) / std::sqrt(x) -
                          out_ * (std::copysign(T{1.0}, x_) + T{1.0} / x_)) *
                         out_grad_);
    }
  }

 private:
  const T* input_x_;
  const T* input_out_;
  const T* input_out_grad_;
  T* output_x_grad_;
  int64_t numel_;
};

}  // namespace phi
