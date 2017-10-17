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

#ifndef HL_ACTIVATION_FUNCTIONS_H_
#define HL_ACTIVATION_FUNCTIONS_H_

#include "hl_functions.h"
#include "paddle/operators/math/lstm_compute.h"

/**
 * Active functions: sigmoid, relu, tanh and linear.
 */
#define FLOAT_ACTIVE_FUNCTION                                   \
  {                                                             \
    hppl::typef::sigmoid, hppl::typef::relu, hppl::typef::tanh, \
        hppl::typef::linear                                     \
  }

#define DOUBLE_ACTIVE_FUNCTION                                  \
  {                                                             \
    hppl::typed::sigmoid, hppl::typed::relu, hppl::typed::tanh, \
        hppl::typed::linear                                     \
  }

#define AVX_ACTIVE_FUNCTION \
  { hppl::sigmoid, hppl::relu, hppl::tanh, hppl::linear }

namespace hppl {

using activation_mode_t = paddle::operators::math::activation_mode_t;

/**
 * Hppl supports sigmoid, relu, tanh, linear active functions
 * for neural networks' forward and backward activation.
 */
template <class T>
class Active {
 public:
  typedef T (*forward)(T);
  typedef T (*backward)(T, T);
};

template <typename T>
struct ForwardActType;

template <>
struct ForwardActType<float> {
  using type = Active<float>::forward;
};

template <>
struct ForwardActType<double> {
  using type = Active<double>::forward;
};

template <typename T>
struct BackwardActType;

template <>
struct BackwardActType<float> {
  using type = Active<float>::backward;
};

template <>
struct BackwardActType<double> {
  using type = Active<double>::backward;
};

#ifdef __NVCC__
namespace gpu {
static __device__ Active<float>::forward forward[] = FLOAT_ACTIVE_FUNCTION;
static __device__ Active<float>::backward backward[] = FLOAT_ACTIVE_FUNCTION;

static __device__ Active<double>::forward forward_d[] = DOUBLE_ACTIVE_FUNCTION;
static __device__ Active<double>::backward backward_d[] =
    DOUBLE_ACTIVE_FUNCTION;

template <typename T>
struct ForwardAct {
  __device__ typename ForwardActType<T>::type operator()(
      activation_mode_t type);
};

template <>
struct ForwardAct<float> {
  __device__ ForwardActType<float>::type operator()(activation_mode_t type) {
    return forward[type];
  }
};

template <>
struct ForwardAct<double> {
  __device__ ForwardActType<double>::type operator()(activation_mode_t type) {
    return forward_d[type];
  }
};

template <typename T>
struct BackwardAct {
  __device__ typename BackwardActType<T>::type operator()(
      activation_mode_t type);
};

template <>
struct BackwardAct<float> {
  __device__ BackwardActType<float>::type operator()(activation_mode_t type) {
    return backward[type];
  }
};

template <>
struct BackwardAct<double> {
  __device__ BackwardActType<double>::type operator()(activation_mode_t type) {
    return backward_d[type];
  }
};

}  // namespace gpu
#else
namespace cpu {
static Active<float>::forward forward[] = FLOAT_ACTIVE_FUNCTION;
static Active<float>::backward backward[] = FLOAT_ACTIVE_FUNCTION;

static Active<double>::forward forward_d[] = DOUBLE_ACTIVE_FUNCTION;
static Active<double>::backward backward_d[] = DOUBLE_ACTIVE_FUNCTION;

template <typename T>
struct ForwardAct {
  typename ForwardActType<T>::type operator()(activation_mode_t type);
};

template <>
struct ForwardAct<float> {
  ForwardActType<float>::type operator()(activation_mode_t type) {
    return forward[type];
  }
};

template <>
struct ForwardAct<double> {
  ForwardActType<double>::type operator()(activation_mode_t type) {
    return forward_d[type];
  }
};

template <typename T>
struct BackwardAct {
  typename BackwardActType<T>::type operator()(activation_mode_t type);
};

template <>
struct BackwardAct<float> {
  BackwardActType<float>::type operator()(activation_mode_t type) {
    return backward[type];
  }
};

template <>
struct BackwardAct<double> {
  BackwardActType<double>::type operator()(activation_mode_t type) {
    return backward_d[type];
  }
};

}  // namespace cpu

#ifdef __AVX__
namespace avx {
static Active<__m256>::forward forward[] = AVX_ACTIVE_FUNCTION;
static Active<__m256>::backward backward[] = AVX_ACTIVE_FUNCTION;
}  // namespace avx
#endif
#endif

}  // namespace hppl

#endif  // HL_ACTIVATION_FUNCTIONS_H_
