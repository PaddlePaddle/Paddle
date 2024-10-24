// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/function_prototype.h"
#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"

using cinn::common::bfloat16;

CINN_REGISTER_HELPER(cuda_intrinsics_bfloat16) {
  auto target = cinn::common::DefaultNVGPUTarget();
  using cinn::backends::FunctionProto;

// bfloat16
#define REGISTER_EXTERN_FUNC_2_IN_1_BF16(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(        \
      cinn_nvgpu_##func__##_bf16, target, bfloat16, bfloat16, bfloat16);

  REGISTER_EXTERN_FUNC_2_IN_1_BF16(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_BF16(mod)

#undef REGISTER_EXTERN_FUNC_2_IN_1_BF16

#define REGISTER_EXTERN_FUNC_1_IN_1_BF16(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(        \
      cinn_nvgpu_##func__##_bf16, target, bfloat16, bfloat16);

  REGISTER_EXTERN_FUNC_1_IN_1_BF16(ceil)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(floor)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(round)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(trunc)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(sin)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(cos)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(tan)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(exp)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(log)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(log2)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(log10)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(sqrt)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(rsqrt)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(cbrt)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(abs)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(erf)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(sinh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(cosh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(tanh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(asin)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(acos)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(atan)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(asinh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(acosh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(atanh)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16(sigmoid)

#undef REGISTER_EXTERN_FUNC_1_IN_1_BF16

#define REGISTER_EXTERN_FUNC_1_IN_1_BF16_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(                 \
      cinn_nvgpu_##func__##_bf16, target, bfloat16, bool);

  REGISTER_EXTERN_FUNC_1_IN_1_BF16_OUT_BOOL(isnan)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16_OUT_BOOL(isinf)
  REGISTER_EXTERN_FUNC_1_IN_1_BF16_OUT_BOOL(isfinite)

#undef REGISTER_EXTERN_FUNC_1_IN_1_BF16_OUT_BOOL

  return true;
}
