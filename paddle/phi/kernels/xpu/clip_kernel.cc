// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/clip_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ClipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const Scalar& min,
                const Scalar& max,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  using XPUDataType = typename XPUTypeTrait<T>::Type;
  using XPUTypeFP16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
  auto x_data = reinterpret_cast<const XPUDataType*>(x.data<T>());
  auto out_data = reinterpret_cast<XPUDataType*>(out->data<T>());
  if (!std::is_same<phi::dtype::float16, T>::value) {
    float min_fp32_cpu = min.to<float>();
    float max_fp32_cpu = max.to<float>();
    int r = -1;
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    float* min_xpu = RAII_GUARD.alloc_l3_or_gm<float>(1);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       min_xpu,
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(&min_fp32_cpu),
                       sizeof(float));
    float* max_xpu = RAII_GUARD.alloc_l3_or_gm<float>(1);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       max_xpu,
                       phi::CPUPlace(),
                       reinterpret_cast<void*>(&max_fp32_cpu),
                       sizeof(float));
    float16* min_fp16 = RAII_GUARD.alloc_l3_or_gm<float16>(1);
    PADDLE_ENFORCE_XDNN_NOT_NULL(min_fp16);
    float16* max_fp16 = RAII_GUARD.alloc_l3_or_gm<float16>(1);
    PADDLE_ENFORCE_XDNN_NOT_NULL(max_fp16);
    r = xpu::cast_v2<float, float16>(dev_ctx.x_context(),
                                     reinterpret_cast<const float*>(min_xpu),
                                     min_fp16,
                                     1);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        errors::External("XPU cast kernel return wrong value[%d %s].",
                         r,
                         XPUAPIErrorMsg[r]));
    r = xpu::cast_v2<float, float16>(dev_ctx.x_context(),
                                     reinterpret_cast<const float*>(max_xpu),
                                     max_fp16,
                                     1);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        errors::External("XPU cast kernel return wrong value[%d %s].",
                         r,
                         XPUAPIErrorMsg[r]));

    r = xpu::clip_v2(dev_ctx.x_context(),
                     x_data,
                     out_data,
                     x.numel(),
                     static_cast<XPUTypeFP16>(*min_fp16),
                     static_cast<XPUTypeFP16>(*max_fp16));
    PADDLE_ENFORCE_EQ(r,
                      XPU_SUCCESS,
                      phi::errors::External("XPU API(clip_v2) return wrong "
                                            "value[%d %s]",
                                            r,
                                            XPUAPIErrorMsg[r]));
  } else {
    int r = xpu::clip_v2(dev_ctx.x_context(),
                         x_data,
                         out_data,
                         x.numel(),
                         min.to<XPUDataType>(),
                         max.to<XPUDataType>());
    PADDLE_ENFORCE_EQ(r,
                      XPU_SUCCESS,
                      phi::errors::External("XPU API(clip_v2) return wrong "
                                            "value[%d %s]",
                                            r,
                                            XPUAPIErrorMsg[r]));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(clip,
                   XPU,
                   ALL_LAYOUT,
                   phi::ClipKernel,
                   float,
                   phi::dtype::float16,
                   int64_t,
                   int) {}
