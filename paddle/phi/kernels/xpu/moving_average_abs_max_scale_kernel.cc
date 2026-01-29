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
void MovingAverageAbsMaxScaleKernel(
    const Context &dev_ctx,
    const DenseTensor &x,
    const paddle::optional<DenseTensor> &in_accum_in,
    const paddle::optional<DenseTensor> &in_state_in,
    float moving_rate,
    bool is_test,
    DenseTensor *out,
    DenseTensor *out_scale,
    DenseTensor *out_state,
    DenseTensor *out_accum) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto *in = &x;

  if (out != nullptr) {
    dev_ctx.template Alloc<T>(out);
    phi::Copy(dev_ctx, *in, dev_ctx.GetPlace(), false, out);
  }

  // testing
  if (is_test) {
    return;
  }

  // training
  float cur_scale_data = GetAbsMax<XPUType, Context>(
      dev_ctx, reinterpret_cast<const XPUType *>(x.data<T>()), x.numel());

  dev_ctx.template Alloc<T>(out_state);
  dev_ctx.template Alloc<T>(out_accum);
  dev_ctx.template Alloc<T>(out_scale);

  FindMovingAverageAbsMaxFunctor<XPUType, Context>(
      dev_ctx,
      reinterpret_cast<const XPUType *>(in_accum_in.get().data<T>()),
      reinterpret_cast<const XPUType *>(in_state_in.get().data<T>()),
      cur_scale_data,
      moving_rate,
      reinterpret_cast<XPUType *>(out_state->data<T>()),
      reinterpret_cast<XPUType *>(out_accum->data<T>()),
      reinterpret_cast<XPUType *>(out_scale->data<T>()));
}

}  // namespace phi

PD_REGISTER_KERNEL(moving_average_abs_max_scale,
                   XPU,
                   ALL_LAYOUT,
                   phi::MovingAverageAbsMaxScaleKernel,
                   phi::bfloat16,
                   phi::float16,
                   float) {}
