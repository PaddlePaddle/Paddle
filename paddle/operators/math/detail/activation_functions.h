/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <math.h>
#include "paddle/platform/hostdevice.h"

#ifdef __AVX__
#include <immintrin.h>
#endif

namespace paddle {
namespace operators {
namespace math {
namespace detail {

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0

namespace forward {

template <typename T>
DEVICE T linear(const T a) {
  return a;
}

template <typename T>
DEVICE T relu(const T a) {
  return a > static_cast<T>(0.0) ? a : static_cast<T>(0.0);
}

template <typename T>
DEVICE T sigmoid(const T a) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  T tmp = (a < min) ? min : ((a > max) ? max : a);
  return static_cast<T>(1.0) / (static_cast<T>(1.0) + exp(-tmp));
}

template <typename T>
DEVICE T tanh(const T a) {
  T tmp = -2.0 * a;
  tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
  return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

}  // namespace forward

namespace backward {

template <typename T>
DEVICE T linear(const T a, const T b) {
  return a;
}

template <typename T>
DEVICE T relu(const T a, const T b) {
  return a * (b > 0.0 ? 1.0 : 0.0);
}

template <typename T>
DEVICE T sigmoid(const T a, const T b) {
  return a * b * (1.0 - b);
}

template <typename T>
DEVICE T tanh(const T a, const T b) {
  return a * (1.0 - b * b);
}

}  // namespace backward

template <typename T>
struct Active {
  typedef T (*Act)(T);
  typedef T (*ActGrad)(T, T);
};

static DEVICE Active<float>::Act kActFloat[] = {
    &forward::sigmoid<float>, &forward::relu<float>, &forward::tanh<float>,
    &forward::linear<float>};

static DEVICE Active<float>::ActGrad kActGradFloat[] = {
    &backward::sigmoid<float>, &backward::relu<float>, &backward::tanh<float>,
    &backward::linear<float>};

static DEVICE Active<double>::Act kActDouble[] = {
    &forward::sigmoid<double>, &forward::relu<double>, &forward::tanh<double>,
    &forward::linear<double>};

static DEVICE Active<double>::ActGrad kActGradDouble[] = {
    &backward::sigmoid<double>, &backward::relu<double>,
    &backward::tanh<double>, &backward::linear<double>};

namespace forward {
inline DEVICE float activation(float a, int index) {
  return kActFloat[index](a);
}

inline DEVICE double activation(double a, int index) {
  return kActDouble[index](a);
}

}  // namespace forward

namespace backward {
inline DEVICE float activation(float a, float b, int index) {
  return kActGradFloat[index](a, b);
}

inline DEVICE double activation(double a, double b, int index) {
  return kActGradDouble[index](a, b);
}
}  // namespace backward

#ifdef __AVX__
namespace forward {
namespace avx {
__m256 relu(const __m256 a);
__m256 sigmoid(const __m256 a);
__m256 tanh(const __m256 a);
__m256 linear(const __m256 a);
}  // namespace avx
}  // namespace forward

namespace backward {
namespace avx {
__m256 relu(const __m256 a, const __m256 b);
__m256 sigmoid(const __m256 a, const __m256 b);
__m256 tanh(const __m256 a, const __m256 b);
__m256 linear(const __m256 a, const __m256 b);
}  // namespace avx
}  // namespace backward

static Active<__m256>::Act kActAvx[] = {
    &forward::avx::sigmoid, &forward::avx::relu, &forward::avx::tanh,
    &forward::avx::linear};

static Active<__m256>::ActGrad kActGradAvx[] = {
    &backward::avx::sigmoid, &backward::avx::relu, &backward::avx::tanh,
    &backward::avx::linear};

namespace forward {
inline __m256 activation(__m256 a, int index) { return kActAvx[index](a); }
}  // namespace forward

namespace backward {
inline __m256 activation(__m256 a, __m256 b, int index) {
  return kActGradAvx[index](a, b);
}
}  // namespace backward

#endif

}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle
