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
__attribute__((global)) void fast_layer_norm_act_tiny_common(float epsilon,
                                                             int64_t m,
                                                             int64_t n,
                                                             const T* x,
                                                             T* y,
                                                             const float* scale,
                                                             const float* bias,
                                                             float act_param);
template <typename T>
__attribute__((global)) void fast_layer_norm_act_tiny_align32(
    float epsilon,
    int64_t m,
    int64_t n,
    const T* x,
    T* y,
    const float* scale,
    const float* bias,
    float act_param);

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
                       const float* bias,
                       const Activation_t& act) {
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

  float act_param = 0.f;
  if (act.type == api::Activation_t::LEAKY_RELU) {
    act_param = act.leaky_alpha;
  }
  int64_t mxn = m * n;
  for (int64_t i = 0; i < mxn; i++) {
    y[i] = fmax(y[i], y[i] * act_param);
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
                        const float* bias,
                        const Activation_t& act) {
  float act_param = 0.f;
  if (act.type == api::Activation_t::LEAKY_RELU) {
    act_param = act.leaky_alpha;
  }
  if (n <= 832) {
    if (n % 32 == 0 && n < 128) {
      xpu2::plugin::fast_layer_norm_act_tiny_align32<T>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
              eps, m, n, x, y, scale, bias, act_param);
    } else {
      xpu2::plugin::fast_layer_norm_act_tiny_common<T>
          <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
              eps, m, n, x, y, scale, bias, act_param);
    }
  } else {
    layer_norm(ctx, x, y, m, n, eps, scale, bias, nullptr, nullptr);
    leaky_relu(ctx, y, y, m * n, act_param, NULL, NULL);
  }

  return SUCCESS;
}

template <typename T>
int layer_norm_act_fusion(Context* ctx,
                          const T* x,
                          T* y,
                          int64_t m,
                          int64_t n,
                          float eps,
                          const float* scale,
                          const float* bias,
                          const Activation_t& act) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "layer_norm_act_fusion", T);
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
    return cpu_wrapper<T>(ctx, x, y, m, n, eps, scale, bias, act);
  }
  if (ctx->dev().type() == api::kXPU2) {
    return xpu2_wrapper<T>(ctx, x, y, m, n, eps, scale, bias, act);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int layer_norm_act_fusion(Context*,
                                   const float*,
                                   float*,
                                   int64_t,
                                   int64_t,
                                   float,
                                   const float*,
                                   const float*,
                                   const Activation_t& act);
template int layer_norm_act_fusion(Context*,
                                   const float16*,
                                   float16*,
                                   int64_t,
                                   int64_t,
                                   float,
                                   const float*,
                                   const float*,
                                   const Activation_t& act);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
