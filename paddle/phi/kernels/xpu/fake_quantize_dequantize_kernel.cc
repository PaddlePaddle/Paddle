// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/fake_quantize_functor.h"
#include "paddle/phi/kernels/xpu/quantize_dequantize_functor.h"

namespace phi {

template <typename T, typename Context>
void FakeQuantizeDequantizeAbsMaxKernel(const Context &dev_ctx,
                                        const DenseTensor &x,
                                        int bit_length,
                                        int round_type,
                                        DenseTensor *out,
                                        DenseTensor *out_scale) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  int bin_cnt = std::pow(2, bit_length - 1) - 1;
  const XPUType *x_ptr = reinterpret_cast<const XPUType *>(x.data<T>());
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(out_scale);

  float s = GetAbsMax<XPUType, Context>(dev_ctx, x_ptr, x.numel());

  XPUType out_s = static_cast<XPUType>(s);
  XPUType *out_scale_ptr = reinterpret_cast<XPUType *>(out_scale->data<T>());
  int r = xpu::do_host2device(
      dev_ctx.x_context(), &out_s, out_scale_ptr, sizeof(XPUType));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "do_host2device");

  ClipAndFakeQuantDequantFunctor<XPUType, Context>(
      dev_ctx,
      x_ptr,
      s,
      bin_cnt,
      round_type,
      x.numel(),
      reinterpret_cast<XPUType *>(out->data<T>()));
}

template <typename T, typename Context>
void FakeQuantizeDequantizeMovingAverageAbsMaxKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const DenseTensor &in_scale,
    const paddle::optional<DenseTensor> &in_accum,
    const paddle::optional<DenseTensor> &in_state,
    float moving_rate,
    int bit_length,
    bool is_test,
    int round_type,
    DenseTensor *out,
    DenseTensor *out_scale,
    DenseTensor *out_state,
    DenseTensor *out_accum) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(out);
  int bin_cnt = std::pow(2, bit_length - 1) - 1;

  // testing
  if (is_test) {
    T scale;
    memory_utils::Copy(CPUPlace(),
                       reinterpret_cast<void *>(&scale),
                       dev_ctx.GetPlace(),
                       reinterpret_cast<const void *>(in_scale.data<T>()),
                       sizeof(T));
    float scale_fp32 = static_cast<float>(scale);
    ClipAndFakeQuantDequantFunctor<XPUType, Context>(
        dev_ctx,
        reinterpret_cast<const XPUType *>(x.data<T>()),
        scale_fp32,
        bin_cnt,
        round_type,
        x.numel(),
        reinterpret_cast<XPUType *>(out->data<T>()));
    return;
  }

  // training
  float cur_scale_data = GetAbsMax<XPUType, Context>(
      dev_ctx, reinterpret_cast<const XPUType *>(x.data<T>()), x.numel());

  dev_ctx.template Alloc<T>(out_state);
  dev_ctx.template Alloc<T>(out_accum);
  dev_ctx.template Alloc<T>(out_scale);

  float scale_fp32 = FindMovingAverageAbsMaxFunctor<XPUType, Context>(
      dev_ctx,
      reinterpret_cast<const XPUType *>(in_accum.get().data<T>()),
      reinterpret_cast<const XPUType *>(in_state.get().data<T>()),
      cur_scale_data,
      moving_rate,
      reinterpret_cast<XPUType *>(out_state->data<T>()),
      reinterpret_cast<XPUType *>(out_accum->data<T>()),
      reinterpret_cast<XPUType *>(out_scale->data<T>()));

  ClipAndFakeQuantDequantFunctor<XPUType, Context>(
      dev_ctx,
      reinterpret_cast<const XPUType *>(x.data<T>()),
      scale_fp32,
      bin_cnt,
      round_type,
      x.numel(),
      reinterpret_cast<XPUType *>(out->data<T>()));
}

}  // namespace phi

PD_REGISTER_KERNEL(fake_quantize_dequantize_abs_max,
                   XPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeDequantizeAbsMaxKernel,
                   phi::bfloat16,
                   phi::float16,
                   float) {}

PD_REGISTER_KERNEL(fake_quantize_dequantize_moving_average_abs_max,
                   XPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeDequantizeMovingAverageAbsMaxKernel,
                   phi::bfloat16,
                   phi::float16,
                   float) {}
