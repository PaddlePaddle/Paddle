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
/**
 * \file This file implements some intrinsic functions for math operation in
 * host device.
 */
#include "paddle/cinn/runtime/cinn_runtime.h"

extern "C" {

//! math extern functions
//@{
void __cinn_host_tanh_v(const cinn_buffer_t* x, cinn_buffer_t* out);
//@}

inline int cinn_host_find_int(const cinn_buffer_t* buf, int size, int num);

inline int cinn_host_find_float(const cinn_buffer_t* buf, int size, float num);

inline int cinn_host_find_int_nd(
    const cinn_buffer_t* buf, int size, int num, int begin, int stride);

inline int cinn_host_find_float_nd(
    const cinn_buffer_t* buf, int size, float num, int begin, int stride);

#define CINN_HOST_LT_NUM(TYPE_SUFFIX, TYPE)                           \
  inline int cinn_host_lt_num_##TYPE_SUFFIX(const cinn_buffer_t* buf, \
                                            const int size,           \
                                            const TYPE num,           \
                                            const int offset,         \
                                            const int stride);

CINN_HOST_LT_NUM(fp32, float)
CINN_HOST_LT_NUM(fp64, double)
CINN_HOST_LT_NUM(int32, int)
CINN_HOST_LT_NUM(int64, int64_t)

#undef CINN_HOST_LT_NUM

#define CINN_HOST_GT_NUM(TYPE_SUFFIX, TYPE)                           \
  inline int cinn_host_gt_num_##TYPE_SUFFIX(const cinn_buffer_t* buf, \
                                            const int size,           \
                                            const TYPE num,           \
                                            const int offset,         \
                                            const int stride);

CINN_HOST_GT_NUM(fp32, float)
CINN_HOST_GT_NUM(fp64, double)
CINN_HOST_GT_NUM(int32, int)
CINN_HOST_GT_NUM(int64, int64_t)

#undef CINN_HOST_GT_NUM

int cinn_host_resize_bilinear(const cinn_buffer_t* buf,
                              const int c_size,
                              const int in_h,
                              const int in_w,
                              const int out_h,
                              const int out_w,
                              const int n,
                              const int c,
                              const int y,
                              const int x);

int cinn_host_resize_bicubic(const cinn_buffer_t* buf,
                             const int c_size,
                             const int in_h,
                             const int in_w,
                             const int out_h,
                             const int out_w,
                             const int n,
                             const int c,
                             const int y,
                             const int x);

#define FN_INT32(func) cinn_host_##func##_int32

inline int FN_INT32(pow)(int x, int y);

inline int FN_INT32(clz)(int x);

inline int FN_INT32(popc)(int x);

inline int FN_INT32(logical_right_shift)(int x, int y);

#undef FN_INT32

#define FN_INT64(func) cinn_host_##func##_int64

inline int64_t FN_INT64(clz)(int64_t x);

inline int64_t FN_INT64(popc)(int64_t x);

inline int64_t FN_INT64(pow)(int64_t x, int64_t y);

inline int64_t FN_INT64(logical_right_shift)(int64_t x, int64_t y);

#undef FN_INT64

#define FN_FP32(func) cinn_host_##func##_fp32

inline float FN_FP32(cbrt)(float x);

#undef FN_FP32

#define FN_FP64(func) cinn_host_##func##_fp64

inline double FN_FP64(cbrt)(double x);

#undef FN_FP64
}
