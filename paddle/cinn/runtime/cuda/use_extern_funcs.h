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
#include "paddle/cinn/backends/extern_func_jit_register.h"

#ifdef CINN_WITH_CUDA
CINN_USE_REGISTER(cinn_cuda_host_api)
CINN_USE_REGISTER(cuda_intrinsics)
CINN_USE_REGISTER(cuda_intrinsics_reduce)
CINN_USE_REGISTER(cuda_intrinsics_bfloat16)
CINN_USE_REGISTER(cuda_intrinsics_float16)
#endif
