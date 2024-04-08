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
#include "paddle/cinn/common/bfloat16.h"
#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/float16.h"
#include "paddle/cinn/runtime/custom_function.h"

CINN_REGISTER_HELPER(hip_intrinsics_reduce) {
  auto target = cinn::common::DefaultROCMTarget();
  using cinn::backends::FunctionProto;

#define EXPAND_REDUCE_INT32_REGISTER_MARCO(MARCO, ...) \
  MARCO(sum_int32, int, ##__VA_ARGS__)                 \
  MARCO(prod_int32, int, ##__VA_ARGS__)                \
  MARCO(max_int32, int, ##__VA_ARGS__)                 \
  MARCO(min_int32, int, ##__VA_ARGS__)

#define EXPAND_REDUCE_INT64_REGISTER_MARCO(MARCO, ...) \
  MARCO(sum_int64, int64_t, ##__VA_ARGS__)             \
  MARCO(prod_int64, int64_t, ##__VA_ARGS__)            \
  MARCO(max_int64, int64_t, ##__VA_ARGS__)             \
  MARCO(min_int64, int64_t, ##__VA_ARGS__)

#define EXPAND_REDUCE_FP32_REGISTER_MACRO(MACRO, ...) \
  MACRO(sum_fp32, float, ##__VA_ARGS__)               \
  MACRO(prod_fp32, float, ##__VA_ARGS__)              \
  MACRO(max_fp32, float, ##__VA_ARGS__)               \
  MACRO(min_fp32, float, ##__VA_ARGS__)

#define EXPAND_REDUCE_BOOL_REGISTER_MACRO(MACRO, ...) \
  MACRO(all, bool, ##__VA_ARGS__)                     \
  MACRO(any, bool, ##__VA_ARGS__)

#define EXPAND_REDUCE_BF16_REGISTER_MACRO(MACRO, ...) \
  MACRO(sum_bf16, bfloat16, ##__VA_ARGS__)            \
  MACRO(prod_bf16, bfloat16, ##__VA_ARGS__)           \
  MACRO(max_bf16, bfloat16, ##__VA_ARGS__)            \
  MACRO(min_bf16, bfloat16, ##__VA_ARGS__)

#define EXPAND_REDUCE_FP16_REGISTER_MACRO(MACRO, ...) \
  MACRO(sum_fp16, float16, ##__VA_ARGS__)             \
  MACRO(prod_fp16, float16, ##__VA_ARGS__)            \
  MACRO(max_fp16, float16, ##__VA_ARGS__)             \
  MACRO(min_fp16, float16, ##__VA_ARGS__)

#define EXPAND_REDUCE_FP64_REGISTER_MACRO(MACRO, ...) \
  MACRO(sum_fp64, double, ##__VA_ARGS__)              \
  MACRO(prod_fp64, double, ##__VA_ARGS__)             \
  MACRO(max_fp64, double, ##__VA_ARGS__)              \
  MACRO(min_fp64, double, ##__VA_ARGS__)

#define REGISTER_WARP_REDUCE_FUNC_IMPL(REDUCE_TYPE, DTYPE)                   \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_warp_reduce_##REDUCE_TYPE, target) \
      .SetRetType<DTYPE>()                                                   \
      .AddInputType<cinn_buffer_t *>()                                       \
      .AddInputType<int>()                                                   \
      .AddInputType<int>()                                                   \
      .End();
  EXPAND_REDUCE_INT32_REGISTER_MARCO(REGISTER_WARP_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_INT64_REGISTER_MARCO(REGISTER_WARP_REDUCE_FUNC_IMPL)
  // EXPAND_REDUCE_BF16_REGISTER_MACRO(REGISTER_WARP_REDUCE_FUNC_IMPL)
  // EXPAND_REDUCE_FP16_REGISTER_MACRO(REGISTER_WARP_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_FP32_REGISTER_MACRO(REGISTER_WARP_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_FP64_REGISTER_MACRO(REGISTER_WARP_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_BOOL_REGISTER_MACRO(REGISTER_WARP_REDUCE_FUNC_IMPL)
#undef REGISTER_WARP_REDUCE_FUNC_IMPL

  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_warp_reduce_avg_fp32, target)
      .SetRetType<float>()
      .AddInputType<cinn_buffer_t *>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

#define REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL(REDUCE_TYPE, DTYPE) \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(                                \
      cinn_block_reduce_##REDUCE_TYPE##_internal, target)            \
      .SetRetType<DTYPE>()                                           \
      .AddInputType<DTYPE>()                                         \
      .End();

  EXPAND_REDUCE_INT32_REGISTER_MARCO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  EXPAND_REDUCE_INT64_REGISTER_MARCO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  // EXPAND_REDUCE_BF16_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  // EXPAND_REDUCE_FP16_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  EXPAND_REDUCE_FP32_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  EXPAND_REDUCE_FP64_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  EXPAND_REDUCE_BOOL_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)

#undef REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL

#define REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL(REDUCE_TYPE, DTYPE) \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(                                \
      cinn_block_reduce_##REDUCE_TYPE##_internal_shm, target)        \
      .SetRetType<DTYPE>()                                           \
      .AddInputType<DTYPE>()                                         \
      .AddInputType<cinn_buffer_t *>()                               \
      .End();

  EXPAND_REDUCE_INT32_REGISTER_MARCO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  EXPAND_REDUCE_INT64_REGISTER_MARCO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  // EXPAND_REDUCE_BF16_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  // EXPAND_REDUCE_FP16_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  EXPAND_REDUCE_FP32_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  EXPAND_REDUCE_FP64_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)
  EXPAND_REDUCE_BOOL_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL)

#undef REGISTER_BLOCK_REDUCE_INTERNAL_FUNC_IMPL

#define REGISTER_BLOCK_REDUCE_FUNC_IMPL(REDUCE_TYPE, DTYPE)                   \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(cinn_block_reduce_##REDUCE_TYPE, target) \
      .SetRetType<DTYPE>()                                                    \
      .AddInputType<cinn_buffer_t *>()                                        \
      .AddInputType<int>()                                                    \
      .AddInputType<int>()                                                    \
      .End();

  EXPAND_REDUCE_INT32_REGISTER_MARCO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_INT64_REGISTER_MARCO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  // EXPAND_REDUCE_BF16_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  // EXPAND_REDUCE_FP16_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_FP32_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_FP64_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)
  EXPAND_REDUCE_BOOL_REGISTER_MACRO(REGISTER_BLOCK_REDUCE_FUNC_IMPL)

#undef REGISTER_BLOCK_REDUCE_FUNC_IMPL

#define REGISTER_BLOCK_SHUFLLE_FUNC_IMPL(REDUCE_TYPE, DTYPE)              \
  REGISTER_FACKED_EXTERN_FUNC_HELPER(block_shuffle_##REDUCE_TYPE, target) \
      .SetRetType<DTYPE>()                                                \
      .AddInputType<cinn_buffer_t *>()                                    \
      .AddInputType<int>()                                                \
      .End();

  EXPAND_REDUCE_INT32_REGISTER_MARCO(REGISTER_BLOCK_SHUFLLE_FUNC_IMPL)
  EXPAND_REDUCE_INT64_REGISTER_MARCO(REGISTER_BLOCK_SHUFLLE_FUNC_IMPL)
  // EXPAND_REDUCE_BF16_REGISTER_MACRO(REGISTER_BLOCK_SHUFLLE_FUNC_IMPL)
  // EXPAND_REDUCE_FP16_REGISTER_MACRO(REGISTER_BLOCK_SHUFLLE_FUNC_IMPL)
  EXPAND_REDUCE_FP32_REGISTER_MACRO(REGISTER_BLOCK_SHUFLLE_FUNC_IMPL)
  EXPAND_REDUCE_FP64_REGISTER_MACRO(REGISTER_BLOCK_SHUFLLE_FUNC_IMPL)
  EXPAND_REDUCE_BOOL_REGISTER_MACRO(REGISTER_BLOCK_SHUFLLE_FUNC_IMPL)

#undef REGISTER_BLOCK_SHUFLLE_FUNC_IMPL

#undef EXPAND_REDUCE_INT32_REGISTER_MARCO
#undef EXPAND_REDUCE_INT64_REGISTER_MARCO
#undef EXPAND_REDUCE_BF16_REGISTER_MACRO
#undef EXPAND_REDUCE_FP16_REGISTER_MACRO
#undef EXPAND_REDUCE_FP32_REGISTER_MACRO
#undef EXPAND_REDUCE_FP64_REGISTER_MACRO
#undef EXPAND_REDUCE_BOOL_REGISTER_MACRO

  return true;
}
