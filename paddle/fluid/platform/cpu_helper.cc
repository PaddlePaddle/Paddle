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

#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/enforce.h"

#ifdef PADDLE_WITH_MKLML
#include <omp.h>
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#endif

DEFINE_int32(paddle_num_threads, 1,
             "Number of threads for each paddle instance.");

namespace paddle {
namespace platform {

void SetNumThreads(int num_threads) {
#ifdef PADDLE_USE_OPENBLAS
  int real_num_threads = num_threads > 1 ? num_threads : 1;
  openblas_set_num_threads(real_num_threads);
#elif defined(PADDLE_WITH_MKLML)
  int real_num_threads = num_threads > 1 ? num_threads : 1;
  platform::dynload::MKL_Set_Num_Threads(real_num_threads);
  omp_set_num_threads(num_threads);
#else
  PADDLE_ENFORCE(false, "To be implemented.");
#endif
}

}  // namespace platform
}  // namespace paddle
