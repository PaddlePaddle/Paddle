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
#if defined(PADDLE_WITH_XPU_XFT)
#include <xft/xdnn_plugin.h>
#endif
#include "paddle/common/enforce.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void WeightQuantizeKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const std::string& algo,
                          const int32_t arch,
                          const int32_t group_size,
                          DenseTensor* out,
                          DenseTensor* scale) {
#if defined(PADDLE_WITH_XPU_XFT)
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto xpu_ctx = static_cast<const phi::XPUContext*>(&dev_ctx);
  int k = x.dims()[0];
  int n = x.dims()[1];
  scale->Resize({static_cast<int64_t>(n)});

  dev_ctx.template Alloc<float>(scale);
  if (out->numel() == 0) {
    dev_ctx.template Alloc<int8_t>(out);
    return;
  }

  if (algo == "weight_only_int8") {
    out->Resize({static_cast<int64_t>(k), static_cast<int64_t>(n)});
    dev_ctx.template Alloc<int8_t>(out);

    int ret = baidu::xpu::xftkernel::xft_quant2d_per_channel<XPUType, float>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType*>(x.template data<T>()),
        nullptr,
        out->data<int8_t>(),
        scale->data<float>(),
        k,
        n);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "quant2d");
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Weight quantize only supports weight_only_int8 on XPU now."));
  }
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "weight_quantize is not supported since it's not "
      "compiled with XPU_XFT"));
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(weight_quantize,
                   XPU,
                   ALL_LAYOUT,
                   phi::WeightQuantizeKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
