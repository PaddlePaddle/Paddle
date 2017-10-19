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

/**
 * The maximum input value for exp, used to avoid overflow problem.
 * currently only used for tanh function.
 */
#define EXP_MAX_INPUT 40.0

#ifndef __NVCC__
namespace hppl {
namespace typef {
float relu(const float a);
float sigmoid(const float a);
float tanh(const float a);
float linear(const float a);

float relu(const float a, const float b);
float sigmoid(const float a, const float b);
float tanh(const float a, const float b);
float linear(const float a, const float b);

}  // namespace typef

namespace typed {
double relu(const double a);
double sigmoid(const double a);
double tanh(const double a);
double linear(const double a);

double relu(const double a, const double b);
double sigmoid(const double a, const double b);
double tanh(const double a, const double b);
double linear(const double a, const double b);
}  // namespace typed

}  // namespace hppl

#ifdef __AVX__
#include "hl_avx_functions.h"
#endif

#else
#include "hl_gpu_functions.h"
#endif

#endif  // HL_FUNCTIONS_H_
