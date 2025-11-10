/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/fake_quantize_functor.h"

namespace phi {

template <typename T, typename Context>
float GetAbsMax(const Context& dev_ctx, const T* input, int64_t numel) {
  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
  int max_ptr_size = phi::backends::xpu::get_xpu_max_ptr_size(-1);
  float* buffer_for_findmax = RAII_GUARD.alloc_l3_or_gm<float>(max_ptr_size);
  PADDLE_ENFORCE_XDNN_NOT_NULL(buffer_for_findmax);
  std::vector<float> buffer_cpu(max_ptr_size);
  int r =
      xpu::findmax<T>(dev_ctx.x_context(), input, buffer_for_findmax, numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "findmax");
  memory_utils::Copy(CPUPlace(),
                     reinterpret_cast<void*>(buffer_cpu.data()),
                     dev_ctx.GetPlace(),
                     reinterpret_cast<void*>(buffer_for_findmax),
                     sizeof(float) * max_ptr_size);
  return *std::max_element(buffer_cpu.begin(), buffer_cpu.end());
}

template <typename T, typename Context>
void ClipAndFakeQuantDequantFunctor(const Context& dev_ctx,
                                    const T* x_ptr,
                                    const float scale,
                                    const int bin_cnt,
                                    int round_type,
                                    int64_t x_len,
                                    T* out_ptr) {
  float inv_scale = phi::funcs::inverse(scale);
  if (round_type == 0) {
    PADDLE_THROW(common::errors::Unimplemented(
        "round_type == 0 not support in fake_quantize_dequantize_abs_max for "
        "xpu."));
  } else {
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    float* x_fp32 = RAII_GUARD.alloc_l3_or_gm<float>(x_len);
    PADDLE_ENFORCE_XDNN_NOT_NULL(x_fp32);
    int r = xpu::cast<T, float>(dev_ctx.x_context(), x_ptr, x_fp32, x_len);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    r = xpu::clamp<float>(
        dev_ctx.x_context(), x_fp32, x_fp32, x_len, -scale, scale);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "clamp");
    // bin_cnt * inv_s * x
    r = xpu::scale<float>(dev_ctx.x_context(),
                          x_fp32,
                          x_fp32,
                          x_len,
                          false,
                          bin_cnt * inv_scale,
                          0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    // round(bin_cnt * inv_s * x)
    r = xpu::paddle_round<float>(dev_ctx.x_context(), x_fp32, x_fp32, x_len, 0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_round");
    // round(bin_cnt * inv_s * x) * s
    r = xpu::scale<float>(
        dev_ctx.x_context(), x_fp32, x_fp32, x_len, false, scale, 0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    // round(bin_cnt * inv_s * x) * s / bin_cnt
    r = xpu::scale<float>(dev_ctx.x_context(),
                          x_fp32,
                          x_fp32,
                          x_len,
                          false,
                          1.0f / bin_cnt,
                          0.0f);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    r = xpu::cast<float, T>(dev_ctx.x_context(), x_fp32, out_ptr, x_len);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  }
}

template <typename T, typename Context>
float FindMovingAverageAbsMaxFunctor(const Context& dev_ctx,
                                     const T* in_accum,
                                     const T* in_state,
                                     const float cur_scale,
                                     const float rate,
                                     T* out_state,
                                     T* out_accum,
                                     T* out_scale) {
  T accum;
  T state;

  memory_utils::Copy(CPUPlace(),
                     reinterpret_cast<void*>(&accum),
                     dev_ctx.GetPlace(),
                     in_accum,
                     sizeof(T));
  memory_utils::Copy(CPUPlace(),
                     reinterpret_cast<void*>(&state),
                     dev_ctx.GetPlace(),
                     in_state,
                     sizeof(T));
  float accum_fp32 = static_cast<float>(accum);
  float state_fp32 = static_cast<float>(state);

  state_fp32 = rate * state_fp32 + 1;
  accum_fp32 = rate * accum_fp32 + cur_scale;
  float scale_fp32 = accum_fp32 / state_fp32;

  state = static_cast<T>(state_fp32);
  accum = static_cast<T>(accum_fp32);
  T scale = static_cast<T>(scale_fp32);
  int r =
      xpu::do_host2device(dev_ctx.x_context(), &state, out_state, sizeof(T));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "do_host2device state");
  r = xpu::do_host2device(dev_ctx.x_context(), &accum, out_accum, sizeof(T));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "do_host2device accum");
  r = xpu::do_host2device(dev_ctx.x_context(), &scale, out_scale, sizeof(T));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "do_host2device scale");
  return scale_fp32;
}

}  // namespace phi
