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

namespace xpu2 {
namespace plugin {
template <typename T>
__attribute__((global)) void fast_layer_norm_tiny_common(float epsilon,
                                                         int64_t m,
                                                         int64_t n,
                                                         const T* x,
                                                         T* y,
                                                         const float* scale,
                                                         const float* bias);
template <typename T>
__attribute__((global)) void fast_layer_norm_tiny_align32(float epsilon,
                                                          int64_t m,
                                                          int64_t n,
                                                          const T* x,
                                                          T* y,
                                                          const float* scale,
                                                          const float* bias);
}  // namespace plugin
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
static int cpu_wrapper(Context* ctx,
                       const T* x,
                       T* y,
                       int64_t m,
                       int64_t n,
                       float eps,
                       const float* scale,
                       const float* bias) {
  for (int64_t i = 0; i < m; i++) {
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int64_t j = 0; j < n; j++) {
      float v = static_cast<float>(x[i * n + j]);
      sum += v;
      square_sum += v * v;
    }
    float mean_value = sum / n;
    float var_value = square_sum / n - mean_value * mean_value;
    float rstd = 1.0f / std::sqrt(var_value + eps);
    for (int64_t j = 0; j < n; j++) {
      float v = static_cast<float>(x[i * n + j]);
      float scale_value = ((scale == nullptr) ? 1.0f : scale[j]);
      float bias_value = ((bias == nullptr) ? 0.0f : bias[j]);
      float out = (v - mean_value) * rstd * scale_value + bias_value;
      y[i * n + j] = static_cast<T>(out);
    }
  }
  return SUCCESS;
}

template <typename T>
static int xpu2_wrapper(Context* ctx,
                        const T* x,
                        T* y,
                        int64_t m,
                        int64_t n,
                        float eps,
                        const float* scale,
                        const float* bias) {
  if (n <= 832) {
    if (n % 32 == 0 && n < 128) {
      xpu2::plugin::fast_layer_norm_tiny_align32<T>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
              eps, m, n, x, y, scale, bias);
    } else {
      xpu2::plugin::fast_layer_norm_tiny_common<T>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
              eps, m, n, x, y, scale, bias);
    }
  } else {
    return layer_norm(ctx, x, y, m, n, eps, scale, bias, nullptr, nullptr);
  }

  return SUCCESS;
}

template <typename T>
int fast_layer_norm(Context* ctx,
                    const T* x,
                    T* y,
                    int64_t m,
                    int64_t n,
                    float eps,
                    const float* scale,
                    const float* bias) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "fast_layer_norm", T);
  WRAPPER_DUMP_PARAM5(ctx, x, y, m, n, eps);
  WRAPPER_DUMP_PARAM2(ctx, scale, bias);
  WRAPPER_DUMP(ctx);
  int64_t xylen = -1;
  WRAPPER_CHECK_SHAPE(ctx, &xylen, {m, n});
  WRAPPER_CHECK_2PTRS(ctx, T, xylen, x, y);
  WRAPPER_ASSERT_GE(ctx, eps, 0);
  WRAPPER_CHECK_PTR_OR_NULL(ctx, float, n, scale);
  WRAPPER_CHECK_PTR_OR_NULL(ctx, float, n, bias);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx, x, y, m, n, eps, scale, bias);
  }
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper<T>(ctx, x, y, m, n, eps, scale, bias);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int fast_layer_norm(Context*,
                             const float*,
                             float*,
                             int64_t,
                             int64_t,
                             float,
                             const float*,
                             const float*);
template int fast_layer_norm(Context*,
                             const float16*,
                             float16*,
                             int64_t,
                             int64_t,
                             float,
                             const float*,
                             const float*);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
