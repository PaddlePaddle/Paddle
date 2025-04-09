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

#include "paddle/phi/core/platform/cpu_helper.h"

#ifdef PADDLE_WITH_MKLML
#include <omp.h>

#include "paddle/phi/backends/dynload/mklml.h"
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#elif PADDLE_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace paddle::platform {

void SetNumThreads(int num_threads) {
#ifdef PADDLE_USE_OPENBLAS
// windows has no support for openblas multi-thread
// please refer to: https://github.com/PaddlePaddle/Paddle/issues/7234
#ifdef _WIN32
  if (num_threads > 1) {
    num_threads = 1;
  }
#endif
  int real_num_threads = num_threads > 1 ? num_threads : 1;
  openblas_set_num_threads(real_num_threads);
#elif defined(PADDLE_WITH_MKLML)
  int real_num_threads = num_threads > 1 ? num_threads : 1;
  phi::dynload::MKL_Set_Num_Threads(real_num_threads);
  omp_set_num_threads(real_num_threads);
#elif defined(PADDLE_USE_REFERENCE_CBLAS)
  // cblas not support multi-thread
  return;
#elif defined(PADDLE_USE_ACCELERATE)
  // not sure about apple's blas
  return;
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "This library (except OPENBLAS, MKLML) is not supported yet, so the"
      "number of threads cannot be set."));
#endif
}

}  // namespace paddle::platform
