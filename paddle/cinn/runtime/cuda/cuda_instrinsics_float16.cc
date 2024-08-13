// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/float16.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"

using cinn::common::float16;

CINN_REGISTER_HELPER(cuda_intrinsics_float16) {
  auto target = cinn::common::DefaultNVGPUTarget();
  using cinn::backends::FunctionProto;

// float16
#define REGISTER_EXTERN_FUNC_2_IN_1_FP16(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_2_IN_1_OUT(        \
      cinn_nvgpu_##func__##_fp16, target, float16, float16, float16);

  REGISTER_EXTERN_FUNC_2_IN_1_FP16(pow)
  REGISTER_EXTERN_FUNC_2_IN_1_FP16(mod)

#undef REGISTER_EXTERN_FUNC_2_IN_1_FP16

#define REGISTER_EXTERN_FUNC_1_IN_1_FP16(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(        \
      cinn_nvgpu_##func__##_fp16, target, float16, float16);

  REGISTER_EXTERN_FUNC_1_IN_1_FP16(ceil)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(floor)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(round)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(trunc)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(sin)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(cos)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(tan)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(exp)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(log)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(log2)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(log10)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(sqrt)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(rsqrt)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(cbrt)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(abs)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(erf)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(sinh)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(cosh)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(tanh)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(asin)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(acos)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(atan)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(asinh)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(acosh)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(atanh)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16(sigmoid)

#undef REGISTER_EXTERN_FUNC_1_IN_1_FP16

#define REGISTER_EXTERN_FUNC_1_IN_1_FP16_OUT_BOOL(func__) \
  REGISTER_EXTERN_SOURCE_FUNC_1_IN_1_OUT(                 \
      cinn_nvgpu_##func__##_fp16, target, float16, bool);

  REGISTER_EXTERN_FUNC_1_IN_1_FP16_OUT_BOOL(isnan)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16_OUT_BOOL(isinf)
  REGISTER_EXTERN_FUNC_1_IN_1_FP16_OUT_BOOL(isfinite)

#undef REGISTER_EXTERN_FUNC_1_IN_1_FP16_OUT_BOOL

#define REGISTER_CINN_NVGPU_GT_NUM(TYPE_SUFFIX, TYPE)                         \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_nvgpu_gt_num_##TYPE_SUFFIX, target) \
      .SetRetType<int>()                                                      \
      .AddInputType<cinn_buffer_t *>()                                        \
      .AddInputType<int>()                                                    \
      .AddInputType<TYPE>()                                                   \
      .AddInputType<int>()                                                    \
      .AddInputType<int>()                                                    \
      .End();

  REGISTER_CINN_NVGPU_GT_NUM(fp16, float16);

#undef REGISTER_CINN_NVGPU_GT_NUM

#define REGISTER_CINN_NVGPU_LT_NUM(TYPE_SUFFIX, TYPE)                         \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_nvgpu_lt_num_##TYPE_SUFFIX, target) \
      .SetRetType<int>()                                                      \
      .AddInputType<cinn_buffer_t *>()                                        \
      .AddInputType<int>()                                                    \
      .AddInputType<TYPE>()                                                   \
      .AddInputType<int>()                                                    \
      .AddInputType<int>()                                                    \
      .End();

  REGISTER_CINN_NVGPU_LT_NUM(fp16, float16);

#undef REGISTER_CINN_NVGPU_LT_NUM

#define REGISTER_CINN_NVGPU_INDEX_ADD(TYPE_SUFFIX, TYPE)                 \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_nvgpu_index_add_##TYPE_SUFFIX, \
                                     target)                             \
      .SetRetType<TYPE>()                                                \
      .AddInputType<TYPE>()                                              \
      .AddInputType<int>()                                               \
      .AddInputType<cinn_buffer_t *>()                                   \
      .AddInputType<int>()                                               \
      .AddInputType<int>()                                               \
      .AddInputType<cinn_buffer_t *>()                                   \
      .AddInputType<int>()                                               \
      .End();

  REGISTER_CINN_NVGPU_INDEX_ADD(fp16, float16);

#undef REGISTER_CINN_NVGPU_INDEX_ADD

  return true;
}
