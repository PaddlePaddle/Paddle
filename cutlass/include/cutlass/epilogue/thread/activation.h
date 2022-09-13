/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief This extends the contents of cutlass/functional.h with frequently used activation functions.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/constants.h"
#include "cutlass/complex.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Identity {
  CUTLASS_HOST_DEVICE
  T operator()(T value) const {
    return value;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// ReLu operator - propagates NaNs
/// Always put threshold in the right hand side of max to propagate NaN.
template <typename T>
struct ReLu {
  static const bool kIsHeavy=false;
  CUTLASS_HOST_DEVICE
  T operator()(T const & threshold, T value) const {
    maximum<T> mx;

    return mx(value, threshold);
  }

  CUTLASS_HOST_DEVICE
  T operator()(T value) const {
    maximum<T> mx;

    return mx(value, T(0));
  }
};

template <typename T, int N>
struct ReLu<Array<T, N>> {
  static const bool kIsHeavy=false;
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const & threshold, Array<T, N> const &frag) const {
    maximum<Array<T, N> > mx;

    return mx(frag, threshold);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &frag) const {
    maximum<Array<T, N> > mx;
    return mx(frag, T(0));
  }
};

// Leaky Relu operator
template <typename T>
struct LeakyReLU {
  CUTLASS_HOST_DEVICE
  T operator()(T const &value, T const & alpha_recip) const {
    T res = value > T(0) ? value : value * alpha_recip;
    return res;
  }
};

template <typename T, int N>
struct LeakyReLU<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs, T const & alpha_recip) const {
    Array<T, N> y;
    LeakyReLU<T> leaky_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(rhs.size()); ++i) {
      y[i] = leaky_op(rhs[i], alpha_recip);
    }

    return y;
  }
};

// Sigmoid operator
template <typename T>
struct Sigmoid {
  CUTLASS_HOST_DEVICE
  T operator()(T const &scalar) const {
    return T(1) / (T(1) + fast_exp(-scalar));
  }
};

template <typename T, int N>
struct Sigmoid<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    Sigmoid<T> sigmoid_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = sigmoid_op(rhs[i]);
    }

    return y;
  }
};

template <int N>
struct Sigmoid<Array<half_t, N>> {
  using T = half_t;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& z) const {
    plus<Array<T, N>> add;

#if defined(CUTLASS_USE_TANH_FOR_SIGMOID)
    multiplies<Array<T, N>> mul;
    fast_tanh_op<Array<T, N>> tanh;
    return mul(add(tanh(mul(z, cutlass::constants::half<T>())), cutlass::constants::one<T>()),
               cutlass::constants::half<T>());
#else
    divides<Array<T, N>> div;
    negate<Array<T, N>> neg;
    fast_exp_op<Array<T, N>> fast_exp;
    return div(cutlass::constants::one<T>(),
               add(cutlass::constants::one<T>(),
                   fast_exp(neg(z))));
#endif
  }
};

// SiLu (swish) operator introduced by Elfwing et al. in the following paper
// "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning" (2017)
// https://arxiv.org/pdf/1702.03118.pdf
// It is used in EfficientNet and YOLOv5, for example.
// Reference: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
template <typename T>
struct SiLu {
  CUTLASS_HOST_DEVICE
  T operator()(T const &scalar) const {
    Sigmoid<T> sigmoid;
    return scalar * sigmoid(scalar);
  }
};

template <typename T, int N>
struct SiLu<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Sigmoid<Array<T, N>> sigmoid_op;
    multiplies<Array<T, N>>     mul;
    return mul(rhs, sigmoid_op(rhs));
  }
};

// Hardswish operator introduced by Howard et al. in the following paper
// "Searching for MobileNetV3" (2019)
// https://arxiv.org/pdf/1905.02244.pdf
// It is used in models based on MobilenetNetV3.
// Reference: https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html
template <typename T>
struct HardSwish {
  CUTLASS_HOST_DEVICE
  T operator()(T const &x) const {
    minimum<T> mn;
    maximum<T> mx;
    T relu6 = mn(mx(x + T(3), T(0)), T(6));
    return x * relu6 / T(6);
  }
};

template <>
struct HardSwish<float> {
  using T = float;

  CUTLASS_HOST_DEVICE
  T operator()(T const &x) const {
    minimum<T> mn;
    maximum<T> mx;
    T relu6 = mn(mx(x + T(3), T(0)), T(6));
    return x * relu6 * 0.16666667f;
  }
};

template <typename T, int N>
struct HardSwish<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    HardSwish<T> hardswish_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = hardswish_op(rhs[i]);
    }

    return y;
  }
};

template <int N>
struct HardSwish<Array<half_t, N> > {
  using T = half_t;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    minimum<Array<T, N> > mn;
    maximum<Array<T, N> > mx;
    multiplies<Array<T, N> > mul;
    plus<Array<T, N> > add;
 
    return mul(mul(mn(mx(add(rhs, T(3)), T(0)), T(6)), rhs), T(0.16666667f));
  }
};

//
// GELU function definitions implemented as described by
//   Hendrycks, D., and Gimpel, K. in
//   "Gaussian Error Linear Units (GELUs)." (2020)
//   https://arxiv.org/pdf/1606.08415.pdf
//
// Floating-point constants are Taylor coefficients described in the paper.
//

// GELU operator
template <typename T>
struct GELU {
  CUTLASS_HOST_DEVICE
  T operator()(T const &scalar) const {
    return T(cutlass::constants::half<T>() * scalar *
      (cutlass::constants::one<T>() + (T)erff((float)(scalar / cutlass::constants::root_two<T>()))));
  }
};

template <>
struct GELU<float> {
  CUTLASS_HOST_DEVICE
  float operator()(float const &scalar) const {
    return cutlass::constants::half<float>() * scalar *
      (cutlass::constants::one<float>() + erff( scalar / cutlass::constants::root_two<float>() ));
  }
};

template <>
struct GELU<double> {
  CUTLASS_HOST_DEVICE
  double operator()(double const &scalar) const {
    return cutlass::constants::half<double>() * scalar *
      (cutlass::constants::one<double>() + erf( scalar / cutlass::constants::root_two<double>() ));
  }
};

template <typename T, int N>
struct GELU<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    GELU<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(rhs[i]);
    }

    return y;
  }
};

// GELU operator implemented using the Taylor series approximation
template <typename T>
struct GELU_taylor {
  static const bool kIsHeavy=true;
  CUTLASS_HOST_DEVICE
  T operator()(T const &z) const {

    T k0 = T(0.7978845608028654);
    T k1 = T(0.044715);

    return T(cutlass::constants::half<T>() * z *
      (cutlass::constants::one<T>() + fast_tanh(k0 * z * (cutlass::constants::one<T>() + k1 * z * z))));
  }
};

template <int N>
struct GELU_taylor<Array<half_t, N> > {
  static const bool kIsHeavy=true;
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &z) const {

    using T = half_t;
    Array<half_t, N> y;

    half_t k0 = half_t(0.7978845608028654);
    half_t k1 = half_t(0.044715);

    multiply_add<Array<half_t, N>> fma;
    multiplies<Array<half_t, N>>     mul;
    plus<Array<half_t, N>>         add;

    fast_tanh_op<Array<half_t, N>> tanh;

    Array<half_t, N> u = mul(mul(k0, z), fma(mul(k1, z), z, cutlass::constants::one<T>()));

    y = mul(mul(z, cutlass::constants::half<T>()), add(cutlass::constants::one<T>(), tanh(u)));

    return y;
  }
};

template <typename T, int N>
struct GELU_taylor<Array<T, N> > {
  static const bool kIsHeavy=true;
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &rhs) const {
    Array<T, N> y;
    GELU_taylor<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(rhs[i]);
    }

    return y;
  }
};

/// Computes backwards pass for GELU operator assuming d_t is the layer gradient and
/// z is computed from the forward pass.
template <typename T>
struct dGELU {
  CUTLASS_HOST_DEVICE
  T operator()(T const &d_t, T const &z) const {

    T k0 = T(0.7978845608028654);
    T k1 = T(0.044715);
    T k2 = T(0.1070322243);

    T tanh_out = fast_tanh(k0 * z * (1 + k1 * z * z));

    T ff = constants::half<T>() * z * ((1 - tanh_out * tanh_out) * (k0 + k2 * z * z)) +
      constants::half<T>() * (1 + tanh_out);

    return ff * d_t;
  }
};

template <typename T, int N>
struct dGELU<Array<T, N> > {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &d_t, Array<T, N> const &z) const {
    Array<T, N> y;
    dGELU<T> gelu_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      y[i] = gelu_op(d_t[i], z[i]);
    }

    return y;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
