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

#ifndef HL_FUNCTIONS_H_
#define HL_FUNCTIONS_H_

/**
 * sigmoid threshold maximum
 */
#define SIGMOID_THRESHOLD_MIN -40.0

/**
 * sigmoid threshold minimum
 */
#define SIGMOID_THRESHOLD_MAX 13.0

#ifndef __NVCC__
namespace hppl {
/*
 * forward activation
 */
template <typename T>
T relu(const T a);
template <typename T>
T sigmoid(const T a);
template <typename T>
T tanh(const T a);
template <typename T>
T linear(const T a);

/*
 * backward activation
 */
template <typename T>
T relu(const T a, const T b);
template <typename T>
T sigmoid(const T a, const T b);
template <typename T>
T tanh(const T a, const T b);
template <typename T>
T linear(const T a, const T b);
}  // namespace hppl

#ifdef __AVX__
#include "hl_avx_functions.h"
#endif

#else
#include "hl_gpu_functions.h"
#endif

#endif  // HL_FUNCTIONS_H_
