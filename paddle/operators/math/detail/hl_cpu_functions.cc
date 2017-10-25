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
namespace typef {

float relu(const float a) {
  return a > static_cast<float>(0.0) ? a : static_cast<float>(0.0);
}

float sigmoid(const float a) {
  const float min = SIGMOID_THRESHOLD_MIN;
  const float max = SIGMOID_THRESHOLD_MAX;
  float tmp = (a < min) ? min : ((a > max) ? max : a);
  return static_cast<float>(1.0) / (static_cast<float>(1.0) + exp(-tmp));
}

float tanh(const float a) {
  float tmp = -2.0 * a;
  tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
  return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

float linear(const float a) { return a; }

float relu(const float a, const float b) { return a * (b > 0.0 ? 1.0 : 0.0); }

float sigmoid(const float a, const float b) {
  return a * b * (static_cast<float>(1) - b);
}

float tanh(const float a, const float b) {
  return a * (static_cast<float>(1) - b * b);
}

float linear(const float a, const float b) { return a; }

}  // namespace typef

namespace typed {
double relu(const double a) {
  return a > static_cast<double>(0.0) ? a : static_cast<double>(0.0);
}

double sigmoid(const double a) {
  const double min = SIGMOID_THRESHOLD_MIN;
  const double max = SIGMOID_THRESHOLD_MAX;
  double tmp = (a < min) ? min : ((a > max) ? max : a);
  return static_cast<double>(1.0) / (static_cast<double>(1.0) + exp(-tmp));
}

double tanh(const double a) {
  double tmp = -2.0 * a;
  tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
  return (2.0 / (1.0 + exp(tmp))) - 1.0;
}

double linear(const double a) { return a; }

double relu(const double a, const double b) {
  return a * (b > 0.0 ? 1.0 : 0.0);
}

double sigmoid(const double a, const double b) {
  return a * b * (static_cast<double>(1) - b);
}

double tanh(const double a, const double b) {
  return a * (static_cast<double>(1) - b * b);
}

double linear(const double a, const double b) { return a; }

}  // namespace typed
}  // namespace hppl
