// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/runtime/cpu/thread_backend.h"

#include <algorithm>
#include <vector>

#ifdef CINN_USE_OPENMP
#include <omp.h>
#endif  // CINN_USE_OPENMP

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/llvm/runtime_symbol_registry.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/common/enforce.h"

int max_concurrency() {
  int max_concurrency = 1;
  const char* val = getenv("CINN_NUM_THREADS");
  if (val == nullptr) {
    val = getenv("OMP_NUM_THREADS");
  }
  if (val != nullptr) {
    max_concurrency = atoi(val);
  } else {
    max_concurrency = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
    max_concurrency /= 2;  // ignore hyper-threading
#endif
  }
  return std::max(max_concurrency, 1);
}

int cinn_backend_parallel_launch(FCINNParallelLambda flambda,
                                 void* datas,
                                 int num_task) {
  int num_workers = max_concurrency();
  if (num_task == 0) num_task = num_workers;
#ifdef CINN_USE_OPENMP
  omp_set_num_threads(num_task);
#pragma omp parallel num_threads(num_task)
  {
    int thread_num = omp_get_thread_num();
    (*flambda)(thread_num, num_task, datas);
  }
#else
  PADDLE_THROW(::common::errors::Fatal(
      "CINN host parallel launch need OpenMP! Please check."));
#endif  // CINN_USE_OPENMP
  return 0;
}

CINN_REGISTER_HELPER(cinn_backend_parallel) {
  using namespace cinn;  // NOLINT
  using backends::FunctionProto;
  auto host_target = cinn::common::DefaultHostTarget();
  backends::GlobalSymbolRegistry::Global().RegisterFn(
      runtime::intrinsic::parallel_launch,
      reinterpret_cast<void*>(&cinn_backend_parallel_launch));
  return true;
}
