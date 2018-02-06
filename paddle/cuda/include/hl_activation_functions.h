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

/**
 * Active functions: sigmoid, relu, tanh and linear.
 */
#define HPPL_ACTIVE_FUNCTION \
  { hppl::sigmoid, hppl::relu, hppl::tanh, hppl::linear }

namespace hppl {

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

#ifdef __NVCC__
namespace gpu {
static __device__ Active<real>::forward forward[] = HPPL_ACTIVE_FUNCTION;
static __device__ Active<real>::backward backward[] = HPPL_ACTIVE_FUNCTION;
}  // namespace gpu
#else
namespace cpu {
static Active<real>::forward forward[] = HPPL_ACTIVE_FUNCTION;
static Active<real>::backward backward[] = HPPL_ACTIVE_FUNCTION;
}  // namespace cpu

#ifdef __AVX__
namespace avx {
static Active<__m256>::forward forward[] = HPPL_ACTIVE_FUNCTION;
static Active<__m256>::backward backward[] = HPPL_ACTIVE_FUNCTION;
}  // namespace avx
#endif
#endif

}  // namespace hppl

#endif  // HL_ACTIVATION_FUNCTIONS_H_
