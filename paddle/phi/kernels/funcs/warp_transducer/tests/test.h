// Copyright 2018-2019, Mingkun Huang
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "paddle/phi/kernels/funcs/warp_transducer/include/detail/type_def.h"
#include "paddle/phi/kernels/funcs/warp_transducer/include/rnnt.h"
#include "random.hpp"

inline void throw_on_error(rnntStatus_t status, const char* message) {
  if (status != RNNT_STATUS_SUCCESS) {
    throw std::runtime_error(
        message + (", stat = " + std::string(rnntGetStatusString(status))));
  }
}

// float* genActs(int size);
// void genActs(std::vector<float>& arr);
// std::vector<int> genLabels(int alphabet_size, int L);

float rel_diff(const std::vector<float>& grad,
               const std::vector<float>& num_grad) {
  float diff = 0.;
  float tot = 0.;
  for (size_t idx = 0; idx < grad.size(); ++idx) {
    diff += (grad[idx] - num_grad[idx]) * (grad[idx] - num_grad[idx]);
    tot += grad[idx] * grad[idx];
  }

  return diff / tot;
}

// Numerically stable softmax for a minibatch of 1
void softmax(const float* const acts,
             int alphabet_size,
             int T,
             float* probs,
             bool applylog) {
  for (int t = 0; t < T; ++t) {
    float max_activation = -std::numeric_limits<float>::infinity();

    for (int a = 0; a < alphabet_size; ++a)
      max_activation = std::max(max_activation, acts[t * alphabet_size + a]);

    float denom = 0;
    for (int a = 0; a < alphabet_size; ++a)
      denom += std::exp(acts[t * alphabet_size + a] - max_activation);

    for (int a = 0; a < alphabet_size; ++a) {
      probs[t * alphabet_size + a] =
          std::exp(acts[t * alphabet_size + a] - max_activation) / denom;
      if (applylog) {
        probs[t * alphabet_size + a] = std::log(probs[t * alphabet_size + a]);
      }
    }
  }
}
