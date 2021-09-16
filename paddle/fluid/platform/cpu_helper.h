/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <stddef.h>

namespace paddle {
namespace platform {

//! Set the number of threads in use.
void SetNumThreads(int num_threads);

//! Get the number of threads outsize the parallel region.
static inline int64_t GetMaxThreads() {
  int64_t num_threads = 1;
#ifdef PADDLE_WITH_MKLML
  // Do not support nested omp parallem.
  num_threads = omp_in_parallel() ? 1 : omp_get_max_threads();
#endif
  return num_threads > 1 ? num_threads : 1;
}

using ThreadHandler =
    std::function<void(const int64_t begin, const int64_t end)>;

//! Run f in parallel.
static inline void RunParallelFor(const int64_t begin, const int64_t end,
                                  const ThreadHandler& f) {
  if (begin >= end) {
    return;
  }

#ifdef PADDLE_WITH_MKLML
  int64_t max_threads = GetMaxThreads();
  int64_t num_threads = max_threads > end - begin ? end - begin : max_threads;
  if (num_threads > 1) {
#pragma omp parallel num_threads(num_threads)
    {
      int64_t tid = omp_get_thread_num();
      int64_t chunk_size = (end - begin + num_threads - 1) / num_threads;
      int64_t begin_tid = begin + tid * chunk_size;
      int64_t end_tid =
          chunk_size + begin_tid > end ? end : chunk_size + begin_tid;
      f(begin_tid, end_tid);
    }
    return;
  }
#endif

  f(begin, end);
}

}  // namespace platform
}  // namespace paddle
