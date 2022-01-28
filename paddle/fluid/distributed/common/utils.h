// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <sys/time.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace distributed {

template <typename T>
inline paddle::operators::math::BlasT<paddle::platform::CPUDeviceContext, T>
GetBlas() {
  paddle::platform::CPUDeviceContext cpu_ctx;
  return paddle::operators::math::GetBlas<paddle::platform::CPUDeviceContext,
                                          T>(cpu_ctx);
}

template <typename T>
inline void SQRT(int n, const T* x, T* z) {
  for (int i = 0; i < n; ++i) {
    z[i] = sqrt(x[i]);
  }
}

template <typename T>
inline void ADD(int n, const T* x, const T y, T* z) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y;
  }
}

template <typename T>
inline void DIV(int n, const T x, const T* y, T* z) {
  for (int i = 0; i < n; ++i) {
    z[i] = x / y[i];
  }
}

template <typename T>
inline void ELE_MUL(int n, const T* x, const T* y, T* z) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
}

static bool StartWith(const std::string& str, const std::string& substr) {
  return str.find(substr) == 0;
}

static bool EndWith(const std::string& str, const std::string& substr) {
  return str.rfind(substr) == (str.length() - substr.length());
}

inline std::vector<int> bucket(const int v_size, const int b_size) {
  int remainder = v_size % b_size;
  int bucket = v_size / b_size;
  std::vector<int> ret_vec(b_size, bucket);
  for (int i = 0; i < remainder; ++i) {
    ret_vec[i] = ret_vec[i] + 1;
  }
  int cur_bucket = 0;
  for (int& j : ret_vec) {
    int tmp = j;
    j = cur_bucket;
    cur_bucket += tmp;
  }
  ret_vec.push_back(cur_bucket);
  return ret_vec;
}

template <typename T>
std::string to_string(const std::vector<T>& vec) {
  std::stringstream ss;
  for (const auto& c : vec) {
    ss << c << " ";
  }
  return ss.str();
}

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

}  // namespace distributed
}  // namespace paddle
