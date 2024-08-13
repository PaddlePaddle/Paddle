// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T>
static void DispatchComputeImpl(const phi::XPUContext *xpu_ctx,
                                const DenseTensor &x,
                                const DenseTensor *bias,
                                const DenseTensor &dequant_scales,
                                const DenseTensor &shift,
                                const DenseTensor &smooth,
                                const std::string &act_method,
                                const float quant_scale,
                                const int quant_round_type,
                                const float quant_max_bound,
                                const float quant_min_bound,
                                DenseTensor *out) {
  PADDLE_THROW(
      common::errors::Unimplemented("fused_bias_act with smooth "
                                    "quant on xpu is not implemented yet."));
}

template <typename T>
static void ComputeImpl(const phi::XPUContext *xpu_ctx,
                        const DenseTensor &x,
                        const paddle::optional<DenseTensor> &bias,
                        const std::string &act_method,
                        DenseTensor *out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  int rows = x.dims()[0];
  int cols = x.dims()[1];
  int r = 0;
  if (bias) {
    r = baidu::xpu::api::broadcast_add<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        reinterpret_cast<const XPUType *>(bias.get().data<T>()),
        reinterpret_cast<XPUType *>(const_cast<T *>(x.data<T>())),
        {rows, cols},
        {1, cols});
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::broadcast_add failed."));
  }
  if (act_method == "geglu") {
    PD_THROW(
        "NOT supported GeGLU. "
        "Currently Only Support SwiGLU, GeLU, ReLU");
  } else if (act_method == "swiglu") {
    r = baidu::xpu::api::swiglu<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        reinterpret_cast<XPUType *>(out->data<T>()),
        {rows, cols},
        1,
        true);
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::swiglu failed."));
  } else if (act_method == "gelu") {
    r = baidu::xpu::api::gelu<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        reinterpret_cast<XPUType *>(out->data<T>()),
        rows * cols);
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::gelu failed."));
  } else if (act_method == "relu") {
    r = baidu::xpu::api::relu<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(x.data<T>()),
        reinterpret_cast<XPUType *>(out->data<T>()),
        rows * cols);
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::relu failed."));
  } else {
    PD_THROW(
        "NOT supported. "
        "Currently Only Support SwiGLU, GeLU, ReLU");
  }
}

template <typename T, typename Context>
void FusedBiasActKernel(const Context &dev_ctx,
                        const DenseTensor &x,
                        const paddle::optional<DenseTensor> &bias,
                        const paddle::optional<DenseTensor> &dequant_scales,
                        const paddle::optional<DenseTensor> &shift,
                        const paddle::optional<DenseTensor> &smooth,
                        const std::string &act_method,
                        const std::string &compute_dtype,
                        float quant_scale,
                        int quant_round_type,
                        float quant_max_bound,
                        float quant_min_bound,
                        DenseTensor *out) {
  auto xpu_ctx = static_cast<const phi::XPUContext *>(&dev_ctx);
  dev_ctx.template Alloc<T>(out);

  if (dequant_scales && dequant_scales.get().numel() > 0) {
    return DispatchComputeImpl<T>(xpu_ctx,
                                  x,
                                  bias ? &(bias.get()) : nullptr,
                                  dequant_scales.get(),
                                  shift.get(),
                                  smooth.get(),
                                  act_method,
                                  quant_scale,
                                  quant_round_type,
                                  quant_max_bound,
                                  quant_min_bound,
                                  out);
  } else {
    return ComputeImpl<T>(xpu_ctx, x, bias, act_method, out);
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_bias_act,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBiasActKernel,
                   float,
                   phi::dtype::float16) {}
