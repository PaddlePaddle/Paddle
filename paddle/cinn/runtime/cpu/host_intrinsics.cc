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

#include "paddle/cinn/runtime/cpu/host_intrinsics.h"

#include <glog/logging.h>
#include <math.h>

#include "paddle/cinn/backends/extern_func_jit_register.h"
#include "paddle/cinn/backends/function_prototype.h"
#include "paddle/cinn/common/target.h"

#ifdef CINN_WITH_MKL_CBLAS
#include "paddle/cinn/runtime/cpu/mkl_math.h"
#endif

extern "C" {

void __cinn_host_tanh_v(const cinn_buffer_t* x, cinn_buffer_t* out) {
  PADDLE_ENFORCE_EQ(
      x->num_elements(),
      out->num_elements(),
      ::common::errors::InvalidArgument(
          "The number of elements in input buffer (x) must be equal to the "
          "number of elements in output buffer (out)."));
  int xn = x->num_elements();
  auto* x_data = reinterpret_cast<float*>(x->memory);
  auto* out_data = reinterpret_cast<float*>(out->memory);
  for (int i = 0; i < x->num_elements(); i++) {
    out_data[i] = tanhf(x_data[i]);
  }
}

#define __cinn_host_find_kernel(buf, size, num, type, begin, stride)     \
  do {                                                                   \
    for (int i = (size - 1) * stride + begin; i >= begin; i -= stride) { \
      if (reinterpret_cast<type*>(buf->memory)[i] == num)                \
        return (i - begin) / stride;                                     \
    }                                                                    \
    return -1;                                                           \
  } while (0)

inline int cinn_host_find_int(const cinn_buffer_t* buf, int size, int num) {
  __cinn_host_find_kernel(buf, size, num, int, 0, 1);
}

inline int cinn_host_find_float(const cinn_buffer_t* buf, int size, float num) {
  __cinn_host_find_kernel(buf, size, num, float, 0, 1);
}

inline int cinn_host_find_int_nd(
    const cinn_buffer_t* buf, int size, int num, int begin, int stride) {
  __cinn_host_find_kernel(buf, size, num, int, begin, stride);
}

inline int cinn_host_find_float_nd(
    const cinn_buffer_t* buf, int size, float num, int begin, int stride) {
  __cinn_host_find_kernel(buf, size, num, float, begin, stride);
}

#undef __cinn_host_find_kernel

inline int cinn_host_next_smallest_int32(
    cinn_buffer_t* buf, int size, int num, int begin, int stride) {
  int id = -1;
  for (int i = begin; i < begin + size * stride; i += stride) {
    if (id == -1 || reinterpret_cast<int*>(buf->memory)[i] <
                        reinterpret_cast<int*>(buf->memory)[id]) {
      id = i;
    }
  }
  if (id != -1) {
    reinterpret_cast<int*>(buf->memory)[id] = 2147483647;
    return (id - begin) / stride;
  }
  return -1;
}

#define CINN_HOST_LT_NUM(TYPE_SUFFIX, TYPE)                                \
  inline int cinn_host_lt_num_##TYPE_SUFFIX(const cinn_buffer_t* buf,      \
                                            const int size,                \
                                            const TYPE num,                \
                                            const int offset,              \
                                            const int stride) {            \
    int out = 0;                                                           \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) { \
      if (reinterpret_cast<TYPE*>(buf->memory)[i] < num) out++;            \
    }                                                                      \
    return out;                                                            \
  }

CINN_HOST_LT_NUM(fp32, float)
CINN_HOST_LT_NUM(fp64, double)
CINN_HOST_LT_NUM(int32, int)
CINN_HOST_LT_NUM(int64, int64_t)

#undef CINN_HOST_LT_NUM

#define CINN_HOST_GT_NUM(TYPE_SUFFIX, TYPE)                                \
  inline int cinn_host_gt_num_##TYPE_SUFFIX(const cinn_buffer_t* buf,      \
                                            const int size,                \
                                            const TYPE num,                \
                                            const int offset,              \
                                            const int stride) {            \
    int out = 0;                                                           \
    for (int i = (size - 1) * stride + offset; i >= offset; i -= stride) { \
      if (reinterpret_cast<TYPE*>(buf->memory)[i] > num) out++;            \
    }                                                                      \
    return out;                                                            \
  }

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
                              const int x) {
  // same with paddle resize when use cv2 backend
  float scale_y = static_cast<float>(in_h) / out_h;
  float scale_x = static_cast<float>(in_w) / out_w;
  float in_y = (y + 0.5F) * scale_y - 0.5F;
  float in_x = (x + 0.5F) * scale_x - 0.5F;
  int in_y_int = static_cast<int>(std::floor(in_y));
  int in_x_int = static_cast<int>(std::floor(in_x));
  float y_lerp = in_y - in_y_int;
  float x_lerp = in_x - in_x_int;
  float p[2][2];

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      int near_y = in_y_int + i;
      int near_x = in_x_int + j;
      near_y = std::max(std::min(near_y, in_h - 1), 0);
      near_x = std::max(std::min(near_x, in_w - 1), 0);
      p[i][j] = reinterpret_cast<int*>(
          buf->memory)[n * c_size * in_h * in_w + c * in_h * in_w +
                       near_y * in_w + near_x];
    }
  }

  float top = p[0][0] * (1.0F - x_lerp) + p[0][1] * x_lerp;
  float bottom = p[1][0] * (1.0F - x_lerp) + p[1][1] * x_lerp;
  float value = top * (1.0F - y_lerp) + bottom * y_lerp;
  return value;
}

int cinn_host_resize_bicubic(const cinn_buffer_t* buf,
                             const int c_size,
                             const int in_h,
                             const int in_w,
                             const int out_h,
                             const int out_w,
                             const int n,
                             const int c,
                             const int y,
                             const int x) {
  // same with paddle resize when use cv2 backend
  float scale_y = static_cast<float>(in_h) / out_h;
  float scale_x = static_cast<float>(in_w) / out_w;
  float in_y = (y + 0.5F) * scale_y - 0.5F;
  float in_x = (x + 0.5F) * scale_x - 0.5F;
  int in_y_int = static_cast<int>(std::floor(in_y));
  int in_x_int = static_cast<int>(std::floor(in_x));
  float y_fract = in_y - std::floor(in_y);
  float x_fract = in_x - std::floor(in_x);
  float p[4][4];

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      int near_y = in_y_int + i - 1;
      int near_x = in_x_int + j - 1;
      near_y = std::max(std::min(near_y, in_h - 1), 0);
      near_x = std::max(std::min(near_x, in_w - 1), 0);
      p[i][j] = reinterpret_cast<int*>(
          buf->memory)[n * c_size * in_h * in_w + c * in_h * in_w +
                       near_y * in_w + near_x];
    }
  }

  float alpha = -0.75F;
  float w[2][4];

  for (int i = 0; i < 2; ++i) {
    float t = (i == 0 ? x_fract : y_fract);
    float t2 = t * t;
    float t3 = t * t * t;
    w[i][0] = alpha * (t3 - 2 * t2 + t);
    w[i][1] = (alpha + 2) * t3 - (3 + alpha) * t2 + 1;
    w[i][2] = -(alpha + 2) * t3 + (3 + 2 * alpha) * t2 - alpha * t;
    w[i][3] = -alpha * t3 + alpha * t2;
  }

  float col[4];

  for (int i = 0; i < 4; ++i) {
    col[i] = 0.0F;
    for (int j = 0; j < 4; ++j) {
      col[i] += p[i][j] * w[0][j];
    }
  }

  float value = 0.0F;

  for (int i = 0; i < 4; ++i) {
    value += col[i] * w[1][i];
  }

  return value;
}

#define FN_FP32(func) cinn_host_##func##_fp32

inline float FN_FP32(cbrt)(float x) { return cbrt(x); }

inline float FN_FP32(pow)(float x, float y) { return powf(x, y); }

#undef FN_FP32

#define FN_FP64(func) cinn_host_##func##_fp64

inline double FN_FP64(cbrt)(double x) { return cbrt(x); }

inline double FN_FP64(pow)(double x, double y) { return pow(x, y); }

#undef FN_FP64

#define FN_INT32(func) cinn_host_##func##_int32

inline int FN_INT32(pow)(int x, int y) {
  if (x == 0 && y < 0) {
    return -1;
  }
  return pow(x, y);
}

inline int FN_INT32(clz)(int x) { return __builtin_clz(x); }

inline int FN_INT32(popc)(int x) { return __builtin_popcount(x); }

inline int FN_INT32(logical_right_shift)(int x, int y) {
  return ((unsigned int)x >> y);
}

#undef FN_INT32

#define FN_INT64(func) cinn_host_##func##_int64

inline int64_t FN_INT64(clz)(int64_t x) { return __builtin_clzll(x); }

inline int64_t FN_INT64(popc)(int64_t x) { return __builtin_popcountll(x); }

inline int64_t FN_INT64(pow)(int64_t x, int64_t y) { return pow(x, y); }

inline int64_t FN_INT64(logical_right_shift)(int64_t x, int64_t y) {
  return ((uint64_t)x >> y);
}

#undef FN_INT64
}  // extern "C"

CINN_REGISTER_HELPER(host_intrinsics) {
  auto host_target = cinn::common::DefaultHostTarget();
  using cinn::backends::FunctionProto;

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(func__) \
  REGISTER_EXTERN_FUNC_1_IN_1_OUT(func__, host_target, float, float);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(erff);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(acosf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(acoshf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(asinf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(asinhf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(atanf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(atanhf);
  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32(cinn_host_cbrt_fp32);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP64(func__) \
  REGISTER_EXTERN_FUNC_1_IN_1_OUT(func__, host_target, double, double);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP64(cinn_host_cbrt_fp64);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP64

#define REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32_INT(func__) \
  REGISTER_EXTERN_FUNC_1_IN_1_OUT(func__, host_target, float, int);

#undef REGISTER_EXTERN_FUNC_1_IN_1_OUT_FP32_INT

#define REGISTER_EXTERN_FUNC_2_IN_1_F(func__) \
  REGISTER_EXTERN_FUNC_2_IN_1_OUT(func__, host_target, float, float, float);

  REGISTER_EXTERN_FUNC_2_IN_1_F(powf)

#undef REGISTER_EXTERN_FUNC_2_IN_1_F

#define REGISTER_EXTERN_FUNC_2_IN_1_FP32(func__) \
  REGISTER_EXTERN_FUNC_2_IN_1_OUT(               \
      cinn_host_##func__##_fp32, host_target, float, float, float);

  REGISTER_EXTERN_FUNC_2_IN_1_FP32(pow)

#undef REGISTER_EXTERN_FUNC_2_IN_1_FP32

#define REGISTER_EXTERN_FUNC_2_IN_1_FP64(func__) \
  REGISTER_EXTERN_FUNC_2_IN_1_OUT(               \
      cinn_host_##func__##_fp64, host_target, double, double, double);

  REGISTER_EXTERN_FUNC_2_IN_1_FP64(pow)

#undef REGISTER_EXTERN_FUNC_2_IN_1_FP64

#define REGISTER_EXTERN_FUNC_2_IN_1_INT32(func__) \
  REGISTER_EXTERN_FUNC_2_IN_1_OUT(                \
      cinn_host_##func__##_int32, host_target, int, int, int);

  REGISTER_EXTERN_FUNC_2_IN_1_INT32(pow)

  REGISTER_EXTERN_FUNC_2_IN_1_INT32(logical_right_shift)

#undef REGISTER_EXTERN_FUNC_2_IN_1_INT32

#define REGISTER_EXTERN_FUNC_2_IN_1_INT64(func__) \
  REGISTER_EXTERN_FUNC_2_IN_1_OUT(                \
      cinn_host_##func__##_int64, host_target, int64_t, int64_t, int64_t);

  REGISTER_EXTERN_FUNC_2_IN_1_INT64(pow)

  REGISTER_EXTERN_FUNC_2_IN_1_INT64(logical_right_shift)

#undef REGISTER_EXTERN_FUNC_2_IN_1_INT64

  REGISTER_EXTERN_FUNC_1_IN_1_OUT(cinn_host_clz_int32, host_target, int, int);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT(
      cinn_host_clz_int64, host_target, int64_t, int64_t);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT(cinn_host_popc_int32, host_target, int, int);

  REGISTER_EXTERN_FUNC_1_IN_1_OUT(
      cinn_host_popc_int64, host_target, int64_t, int64_t);

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_find_int, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_find_float, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<float>()
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_find_int_nd, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_find_float_nd, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<float>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_next_smallest_int32, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

#define _REGISTER_CINN_HOST_LT_NUM(TYPE_SUFFIX, TYPE)                      \
  REGISTER_EXTERN_FUNC_HELPER(cinn_host_lt_num_##TYPE_SUFFIX, host_target) \
      .SetRetType<int>()                                                   \
      .AddInputType<cinn_buffer_t*>()                                      \
      .AddInputType<int>()                                                 \
      .AddInputType<TYPE>()                                                \
      .AddInputType<int>()                                                 \
      .AddInputType<int>()                                                 \
      .End();

  _REGISTER_CINN_HOST_LT_NUM(fp32, float);
  _REGISTER_CINN_HOST_LT_NUM(fp64, double);
  _REGISTER_CINN_HOST_LT_NUM(int32, int);
  _REGISTER_CINN_HOST_LT_NUM(int64, int64_t);

#undef _REGISTER_CINN_HOST_LT_NUM

#define _REGISTER_CINN_HOST_GT_NUM(TYPE_SUFFIX, TYPE)                      \
  REGISTER_EXTERN_FUNC_HELPER(cinn_host_gt_num_##TYPE_SUFFIX, host_target) \
      .SetRetType<int>()                                                   \
      .AddInputType<cinn_buffer_t*>()                                      \
      .AddInputType<int>()                                                 \
      .AddInputType<TYPE>()                                                \
      .AddInputType<int>()                                                 \
      .AddInputType<int>()                                                 \
      .End();

  _REGISTER_CINN_HOST_GT_NUM(fp32, float);
  _REGISTER_CINN_HOST_GT_NUM(fp64, double);
  _REGISTER_CINN_HOST_GT_NUM(int32, int);
  _REGISTER_CINN_HOST_GT_NUM(int64, int64_t);

#undef _REGISTER_CINN_HOST_GT_NUM

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_resize_bilinear, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  REGISTER_EXTERN_FUNC_HELPER(cinn_host_resize_bicubic, host_target)
      .SetRetType<int>()
      .AddInputType<cinn_buffer_t*>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .AddInputType<int>()
      .End();

  return true;
}
