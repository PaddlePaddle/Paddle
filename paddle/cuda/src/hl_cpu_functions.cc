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

#include <math.h>
#include "hl_functions.h"

namespace hppl {

real relu(const real a) { return a > 0.0f ? a : 0.0f; }

real sigmoid(const real a) {
  const real min = SIGMOID_THRESHOLD_MIN;
  const real max = SIGMOID_THRESHOLD_MAX;
  real tmp = (a < min) ? min : ((a > max) ? max : a);
  return 1.0 / (1.0 + exp(-tmp));
}

real tanh(const real a) {
  real tmp = -2.0 * a;
  tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
  return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

real linear(const real a) { return a; }

real relu(const real a, const real b) { return a * (b > 0.0f ? 1.0f : 0.0f); }

real sigmoid(const real a, const real b) { return a * b * (1 - b); }

real tanh(const real a, const real b) { return a * (1.0f - b * b); }

real linear(const real a, const real b) { return a; }
}  // namespace hppl
