// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <ctc.h>

inline void throw_on_error(ctcStatus_t status, const char* message) {
  if (status != CTC_STATUS_SUCCESS) {
    throw std::runtime_error(
        message + (", stat = " + std::string(ctcGetStatusString(status))));
  }
}

#ifdef __CUDACC__
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

inline void throw_on_error(cudaError_t error, const char* message) {
  if (error) {
    throw thrust::system_error(error, thrust::cuda_category(), message);
  }
}

#endif

std::vector<float> genActs(int size) {
  std::vector<float> arr(size);
  std::mt19937 gen(0);
  std::uniform_real_distribution<> dis(0, 1);
  for (int i = 0; i < size; ++i) arr[i] = dis(gen);
  return arr;
}

std::vector<int> genLabels(int alphabet_size, int L) {
  std::vector<int> label(L);

  std::mt19937 gen(1);
  std::uniform_int_distribution<> dis(1, alphabet_size - 1);

  for (int i = 0; i < L; ++i) {
    label[i] = dis(gen);
  }
  // guarantee repeats for testing
  if (L >= 3) {
    label[L / 2] = label[L / 2 + 1];
    label[L / 2 - 1] = label[L / 2];
  }
  return label;
}

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
void softmax(const float* const acts, int alphabet_size, int T, float* probs) {
  for (int t = 0; t < T; ++t) {
    float max_activation = -std::numeric_limits<float>::infinity();

    for (int a = 0; a < alphabet_size; ++a)
      max_activation = std::max(max_activation, acts[t * alphabet_size + a]);

    float denom = 0;
    for (int a = 0; a < alphabet_size; ++a)
      denom += std::exp(acts[t * alphabet_size + a] - max_activation);

    for (int a = 0; a < alphabet_size; ++a)
      probs[t * alphabet_size + a] =
          std::exp(acts[t * alphabet_size + a] - max_activation) / denom;
  }
}
