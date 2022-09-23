// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <assert.h>
#include <time.h>

#include <atomic>
#include <random>

namespace paddle {
namespace distributed {

// Get time in seconds.
inline double current_realtime() {
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_sec + tp.tv_nsec * 1e-9;
}

inline std::default_random_engine& local_random_engine() {
  struct engine_wrapper_t {
    std::default_random_engine engine;
    engine_wrapper_t() {
      static std::atomic<unsigned long> x(0);  // NOLINT
      std::seed_seq sseq = {
          x++, x++, x++, (unsigned long)(current_realtime() * 1000)};  // NOLINT
      engine.seed(sseq);
    }
  };
  thread_local engine_wrapper_t r;
  return r.engine;
}

template <class T = double>
std::uniform_real_distribution<T>& local_uniform_real_distribution() {
  thread_local std::uniform_real_distribution<T> distr;
  assert(distr.a() == 0.0 && distr.b() == 1.0);
  return distr;
}

template <class T = double>
T uniform_real() {
  return local_uniform_real_distribution<T>()(local_random_engine());
}

template <class T = double>
T uniform_real(T a, T b) {
  if (a == b) {
    return a;
  }
  return (T)(a + uniform_real<T>() * (b - a));
}
}  // namespace distributed
}  // namespace paddle
