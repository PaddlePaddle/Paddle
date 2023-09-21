// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
/*
 * copyright (C) 2022 KUNLUNXIN, Inc
 */

#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"
#include "xpu/refactor/util/vector_util.h"

namespace xpu2 {
namespace plugin {
template <typename data_type>
__attribute__((global)) void bn_act_fusion_infer_kernel(float epsilon,
                                                        int64_t img_n,
                                                        int64_t c_start,
                                                        int64_t c_end,
                                                        int64_t img_c,
                                                        int64_t img_h,
                                                        int64_t img_w,
                                                        const data_type* img_gm,
                                                        data_type* out_gm,
                                                        const float* scale_gm,
                                                        const float* bias_gm,
                                                        const float* mean_gm,
                                                        const float* var_gm,
                                                        int act_type);
}  // namespace plugin
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

// CPU implementation
template <typename T>
static int cpu_wrapper(Context* ctx,
                       const T* x,
                       T* y,
                       int64_t n,
                       int64_t c,
                       int64_t h,
                       int64_t w,
                       float eps,
                       const float* scale,
                       const float* bias,
                       const float* global_mean,
                       const float* global_var,
                       bool is_nchw,
                       int act_type) {
  std::vector<T> tmp0(n * c * h * w);
  std::vector<T> tmp1(n * c * h * w);
  int ret = api::SUCCESS;
  if (is_nchw) {
    ret = api::transpose<T>(ctx, x, tmp0.data(), {n, c, h * w}, {1, 0, 2});
  } else {
    ret = api::transpose<T>(ctx, x, tmp0.data(), {n, h * w, c}, {2, 0, 1});
  }
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  int64_t pixels = n * h * w;
  float mean = 0.0f;
  float var = 0.0f;
  float bias_data = 0.0f;
  float scale_data = 0.0f;
  for (int64_t _c = 0; _c < c; _c++) {
    bias_data = bias[_c];
    scale_data = scale[_c];
    if (global_var != nullptr) {
      var = global_var[_c];
      var += eps;
      float temp = ::sqrt(var);
      scale_data = scale_data / temp;
    }
    if (global_mean != nullptr) {
      mean = global_mean[_c];
      bias_data = bias_data - mean * scale_data;
    }
    for (int64_t p = 0; p < pixels; p++) {
      float v = static_cast<float>(tmp0[_c * pixels + p]);
      float result = scale_data * v + bias_data;
      if (act_type == 0) {  // 0 is relu
        result = result > 0 ? result : 0;
      }
      tmp1[_c * pixels + p] = static_cast<T>(result);
    }
  }
  if (is_nchw) {
    return api::transpose<T>(ctx, tmp1.data(), y, {c, n, h * w}, {1, 0, 2});
  } else {
    return api::transpose<T>(ctx, tmp1.data(), y, {c, n, h * w}, {1, 2, 0});
  }
}

// XPU2 implementation
template <typename T>
static int xpu2_wrapper(Context* ctx,
                        const T* x,
                        T* y,
                        int64_t n,
                        int64_t c,
                        int64_t h,
                        int64_t w,
                        float eps,
                        const float* scale,
                        const float* bias,
                        const float* global_mean,
                        const float* global_var,
                        bool is_nchw,
                        int act_type) {
  int64_t tile_c = 2048 * ctx->ncluster();
  if (h == 1 && w == 1 && n > 100 && c <= tile_c) {
    api::ctx_guard RAII_GUARD(ctx);
    T* x_tmp = const_cast<T*>(x);
    T* y_tmp = const_cast<T*>(y);
    x_tmp = RAII_GUARD.alloc<T>(n * c * h * w);
    y_tmp = RAII_GUARD.alloc<T>(n * c * h * w);
    WRAPPER_ASSERT_WORKSPACE(ctx, x_tmp);
    WRAPPER_ASSERT_WORKSPACE(ctx, y_tmp);
    int trans_ret = api::transpose<T>(ctx, x, x_tmp, {n, c}, {1, 0});
    WRAPPER_ASSERT_SUCCESS(ctx, trans_ret);
    xpu2::plugin::bn_act_fusion_infer_kernel<T>
        <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(eps,
                                                   1,
                                                   0,
                                                   c,
                                                   c,
                                                   n,
                                                   1,
                                                   x_tmp,
                                                   y_tmp,
                                                   scale,
                                                   bias,
                                                   global_mean,
                                                   global_var,
                                                   act_type);
    trans_ret = api::transpose<T>(ctx, y_tmp, y, {c, n}, {1, 0});
    WRAPPER_ASSERT_SUCCESS(ctx, trans_ret);
    return api::SUCCESS;
  }
  api::ctx_guard RAII_GUARD(ctx);
  T* x_tmp = const_cast<T*>(x);
  T* y_tmp = const_cast<T*>(y);
  if (!is_nchw) {
    x_tmp = RAII_GUARD.alloc<T>(n * c * h * w);
    y_tmp = RAII_GUARD.alloc<T>(n * c * h * w);
    WRAPPER_ASSERT_WORKSPACE(ctx, x_tmp);
    WRAPPER_ASSERT_WORKSPACE(ctx, y_tmp);
    int trans_ret =
        api::transpose<T>(ctx, x, x_tmp, {n, h, w, c}, {0, 3, 1, 2});
    WRAPPER_ASSERT_SUCCESS(ctx, trans_ret);
  }
  for (int64_t c_start = 0; c_start < c; c_start += tile_c) {
    int64_t c_end = std::min<int64_t>(c_start + tile_c, c);
    xpu2::plugin::bn_act_fusion_infer_kernel<T>
        <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(eps,
                                                   n,
                                                   c_start,
                                                   c_end,
                                                   c,
                                                   h,
                                                   w,
                                                   x_tmp,
                                                   y_tmp,
                                                   scale,
                                                   bias,
                                                   global_mean,
                                                   global_var,
                                                   act_type);
  }
  if (!is_nchw) {
    int trans_ret =
        api::transpose<T>(ctx, y_tmp, y, {n, c, h, w}, {0, 2, 3, 1});
    WRAPPER_ASSERT_SUCCESS(ctx, trans_ret);
  }
  return api::SUCCESS;
}

template <typename T>
int bn_act_fusion_infer(Context* ctx,
                        const T* x,
                        T* y,
                        int64_t n,
                        int64_t c,
                        int64_t h,
                        int64_t w,
                        float eps,
                        const float* scale,
                        const float* bias,
                        const float* global_mean,
                        const float* global_var,
                        bool is_nchw,
                        int act_type) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "bn_act_fusion_infer", T);
  WRAPPER_DUMP_PARAM6(ctx, x, y, n, c, h, w);
  WRAPPER_DUMP_PARAM6(ctx, eps, scale, bias, global_mean, global_var, is_nchw);
  WRAPPER_DUMP_PARAM1(ctx, act_type);
  WRAPPER_DUMP_PARAM1(ctx, ctx->_l3_mgr.get_size());
  WRAPPER_DUMP(ctx);
  WRAPPER_ASSERT_GE(ctx, eps, 0)
  int64_t len = -1;
  WRAPPER_CHECK_SHAPE(ctx, &len, {n, c, h, w});
  WRAPPER_CHECK_2PTRS(ctx, T, len, x, y);
  WRAPPER_CHECK_2PTRS(ctx, float, c, scale, bias);
  WRAPPER_CHECK_PTR_OR_NULL(ctx, float, c, global_mean);
  WRAPPER_CHECK_PTR_OR_NULL(ctx, float, c, global_var);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx,
                          x,
                          y,
                          n,
                          c,
                          h,
                          w,
                          eps,
                          scale,
                          bias,
                          global_mean,
                          global_var,
                          is_nchw,
                          act_type);
  }
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper<T>(ctx,
                           x,
                           y,
                           n,
                           c,
                           h,
                           w,
                           eps,
                           scale,
                           bias,
                           global_mean,
                           global_var,
                           is_nchw,
                           act_type);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int bn_act_fusion_infer(Context*,
                                 const float*,
                                 float*,
                                 int64_t,
                                 int64_t,
                                 int64_t,
                                 int64_t,
                                 float,
                                 const float*,
                                 const float*,
                                 const float*,
                                 const float*,
                                 bool,
                                 int);
template int bn_act_fusion_infer(Context*,
                                 const float16*,
                                 float16*,
                                 int64_t,
                                 int64_t,
                                 int64_t,
                                 int64_t,
                                 float,
                                 const float*,
                                 const float*,
                                 const float*,
                                 const float*,
                                 bool,
                                 int);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
