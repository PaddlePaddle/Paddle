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

#pragma once

#include "cinn/backends/extern_func_jit_register.h"

CINN_USE_REGISTER(host_intrinsics)
#ifdef CINN_WITH_MKL_CBLAS
CINN_USE_REGISTER(mkl_math)
CINN_USE_REGISTER(cinn_cpu_mkl)
#ifdef CINN_WITH_MKLDNN
CINN_USE_REGISTER(cinn_cpu_mkldnn)
#endif
#endif
CINN_USE_REGISTER(cinn_backend_parallel)
