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
#include <xft/xdnn_plugin.h>
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
  using XPUType = typename XPUTypeTrait<T>::Type;
  typedef paddle::float16 data_t;
  auto xpu_ctx = static_cast<const phi::XPUContext*>(&dev_ctx);
  const auto x_transpose = phi::Transpose<phi::dtype::float16, Context>(
      dev_ctx, x, std::vector<int>({1, 0}));
  int m = x_transpose.dims()[0];
  int n = x_transpose.dims()[1];
  scale->Resize({static_cast<int64_t>(m)});

  // // TODO: out 和 scale 的 type 要再看看
  // int ret = baidu::xpu::api::quant2d(
  //     xpu_ctx->x_context(),
  //     reinterpret_cast<const XPUType*>(x.data<data_t>()),
  //     out->data<int8_t>(),
  //     scale->data<float>(),
  //     m,
  //     n);
  // return {
  //     out, scale
  // };
  // typedef paddle::float16 data_t;
  // size_t m = x.dims()[0];
  // size_t n = x.dims()[1];

  // DenseTensor scale_fp32;
  // scale_fp32.Resize({static_cast<int64_t>(m)});
  // dev_ctx.template Alloc<float>(&scale_fp32);

  dev_ctx.template Alloc<float>(scale);
  // auto scale = paddle::full({m}, -1, paddle::DataType::FLOAT32, x.place());
  if (algo == "weight_only_int8") {
    out->Resize({static_cast<int64_t>(m), static_cast<int64_t>(n)});
    dev_ctx.template Alloc<int8_t>(out);
    // int ret = baidu::xpu::api::quant2d(
    //     dev_ctx.x_context(),
    //     reinterpret_cast<const XPUType*>(x.data<T>()),
    //     out->data<int8_t>(),
    //     scale->data<float>(),
    //     m,
    //     n);
    // PADDLE_ENFORCE_XDNN_SUCCESS(ret, "quant2d");

    int ret = baidu::xpu::xftkernel::xft_quant2d<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType*>(x_transpose.template data<data_t>()),
        out->data<int8_t>(),
        scale->data<float>(),
        m,
        n);
    PADDLE_ENFORCE_XDNN_SUCCESS(ret, "quant2d");
  } else if (algo == "weight_only_int4") {
    out->Resize({static_cast<int64_t>(m), static_cast<int64_t>((n + 1) / 2)});
    dev_ctx.template Alloc<int8_t>(out);
    PADDLE_THROW(common::errors::Unavailable(
        "Weight quantize int4 is not supported on XPU now."));
    // int ret =baidu::xpu::api::plugin::quant2d_int4(
    //     xpu_ctx->x_context(),
    //     reinterpret_cast<const XPUType*>(x.data<data_t>()),
    //     out.data<int8_t>(),
    //     scale.data<float>(),
    //     m,
    //     n);
    // return {out, scale};
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Weight quantize only supports weight_only_int8 on XPU now."));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(weight_quantize,
                   XPU,
                   ALL_LAYOUT,
                   phi::WeightQuantizeKernel,
                   phi::dtype::float16) {}
