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

#include "paddle/phi/kernels/fused_bias_act_kernel.h"

#include <type_traits>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
static void ComputeImpl(const Context &dev_ctx,
                        const DenseTensor &x,
                        const optional<DenseTensor> &bias,
                        const std::string &act_method,
                        DenseTensor *out);

template <typename T, typename Context>
static void DispatchInt32ComputeImpl(const Context &dev_ctx,
                                     const DenseTensor &x,
                                     const optional<DenseTensor> &bias,
                                     const DenseTensor &dequant_scales,
                                     const optional<DenseTensor> &shift,
                                     const optional<DenseTensor> &smooth,
                                     const std::string &act_method,
                                     const float quant_scale,
                                     const float quant_max_bound,
                                     DenseTensor *out) {
  DenseTensor compute_x;
  compute_x.Resize(x.dims());
  dev_ctx.template Alloc<T>(&compute_x);

  auto xpu_ctx = static_cast<const XPUContext *>(&dev_ctx);
  using XPUType = typename XPUTypeTrait<T>::Type;
  int64_t cols = x.dims()[x.dims().size() - 1];
  int64_t rows = x.numel() / cols;
  PADDLE_ENFORCE_LE_INT_MAX(rows, "rows");
  PADDLE_ENFORCE_LE_INT_MAX(cols, "cols");

  int r = 0;
  if constexpr (std::is_same_v<T, float>) {
    DenseTensor cast_x;
    cast_x.Resize(x.dims());
    dev_ctx.template Alloc<float>(&cast_x);
    r = baidu::xpu::api::cast<int32_t, float>(dev_ctx.x_context(),
                                              x.data<int32_t>(),
                                              cast_x.data<float>(),
                                              x.numel());
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::cast failed."));
    r = baidu::xpu::api::broadcast_mul<float>(
        dev_ctx.x_context(),
        cast_x.data<float>(),
        dequant_scales.data<float>(),
        compute_x.data<float>(),
        {static_cast<int>(rows), static_cast<int>(cols)},
        {static_cast<int>(cols)});
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::broadcast_mul failed."));
  } else {
    DenseTensor cast_x_fp32;
    cast_x_fp32.Resize(x.dims());
    dev_ctx.template Alloc<float>(&cast_x_fp32);
    DenseTensor compute_x_fp32;
    compute_x_fp32.Resize(x.dims());
    dev_ctx.template Alloc<float>(&compute_x_fp32);
    r = baidu::xpu::api::cast<int32_t, float>(dev_ctx.x_context(),
                                              x.data<int32_t>(),
                                              cast_x_fp32.data<float>(),
                                              x.numel());
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::cast failed."));
    r = baidu::xpu::api::broadcast_mul<float>(
        dev_ctx.x_context(),
        cast_x_fp32.data<float>(),
        dequant_scales.data<float>(),
        compute_x_fp32.data<float>(),
        {static_cast<int>(rows), static_cast<int>(cols)},
        {static_cast<int>(cols)});
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::broadcast_mul failed."));
    r = baidu::xpu::api::cast<float, XPUType>(
        dev_ctx.x_context(),
        compute_x_fp32.data<float>(),
        reinterpret_cast<XPUType *>(compute_x.data<T>()),
        x.numel());
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::cast failed."));
  }

  if (shift || smooth) {
    PADDLE_ENFORCE_EQ(shift && smooth,
                      true,
                      common::errors::InvalidArgument(
                          "shift and smooth must be both set or both unset."));
  }

  DenseTensor compute_out;
  compute_out.Resize(out->dims());
  dev_ctx.template Alloc<T>(&compute_out);
  ComputeImpl<T, Context>(dev_ctx, compute_x, bias, act_method, &compute_out);

  int64_t out_cols = out->dims()[out->dims().size() - 1];
  int64_t out_rows = out->numel() / out_cols;
  PADDLE_ENFORCE_LE_INT_MAX(out_rows, "out_rows");
  PADDLE_ENFORCE_LE_INT_MAX(out_cols, "out_cols");
  if (shift && smooth) {
    r = baidu::xpu::api::broadcast_add<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType *>(compute_out.data<T>()),
        reinterpret_cast<const XPUType *>(shift.get().data<T>()),
        reinterpret_cast<XPUType *>(compute_out.data<T>()),
        {static_cast<int>(out_rows), static_cast<int>(out_cols)},
        {static_cast<int>(out_cols)});
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::broadcast_add failed."));
    r = baidu::xpu::api::broadcast_mul<XPUType>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType *>(compute_out.data<T>()),
        reinterpret_cast<const XPUType *>(smooth.get().data<T>()),
        reinterpret_cast<XPUType *>(compute_out.data<T>()),
        {static_cast<int>(out_rows), static_cast<int>(out_cols)},
        {static_cast<int>(out_cols)});
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::broadcast_mul failed."));
  }

  if (quant_scale > 0) {
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    float *maxptr = RAII_GUARD.template alloc_l3_or_gm<float>(
        dev_ctx.x_context()->max_ptr_size());
    r = baidu::xpu::api::constant<float>(
        dev_ctx.x_context(),
        maxptr,
        dev_ctx.x_context()->max_ptr_size(),
        127.0f / (quant_max_bound * quant_scale));
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::constant failed."));
    r = baidu::xpu::api::quantization<XPUType, int8_t>(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType *>(compute_out.data<T>()),
        out->data<int8_t>(),
        out->numel(),
        maxptr);
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::quantization failed."));
  } else {
    r = baidu::xpu::api::copy(
        dev_ctx.x_context(),
        reinterpret_cast<const XPUType *>(compute_out.data<T>()),
        reinterpret_cast<XPUType *>(out->data<T>()),
        out->numel());
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::copy failed."));
  }
}

template <typename T, typename Context>
static void ComputeImpl(const Context &dev_ctx,
                        const DenseTensor &x,
                        const optional<DenseTensor> &bias,
                        const std::string &act_method,
                        DenseTensor *out) {
  auto xpu_ctx = static_cast<const XPUContext *>(&dev_ctx);
  using XPUType = typename XPUTypeTrait<T>::Type;
  int64_t cols = x.dims()[x.dims().size() - 1];
  int64_t rows = x.numel() / cols;

  // TODO(large-tensor): XPU broadcast_add API not support int64
  PADDLE_ENFORCE_LE_INT_MAX(rows, "rows");
  PADDLE_ENFORCE_LE_INT_MAX(cols, "cols");

  int r = 0;
  const XPUType *x_data = reinterpret_cast<const XPUType *>(x.data<T>());
  DenseTensor bias_out;
  bool use_glu = act_method == "geglu" || act_method == "swiglu";
  if (bias) {
    XPUType *bias_out_data = reinterpret_cast<XPUType *>(out->data<T>());
    if (use_glu) {
      bias_out.Resize(x.dims());
      dev_ctx.template Alloc<T>(&bias_out);
      bias_out_data = reinterpret_cast<XPUType *>(bias_out.data<T>());
    }
    r = baidu::xpu::api::broadcast_add<XPUType>(
        xpu_ctx->x_context(),
        x_data,
        reinterpret_cast<const XPUType *>(bias.get().data<T>()),
        bias_out_data,
        {static_cast<int>(rows), static_cast<int>(cols)},
        {static_cast<int>(cols)});
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::broadcast_add failed."));
    x_data = bias_out_data;
  }
  if (act_method == "geglu") {
    DenseTensor gate;
    gate.Resize(out->dims());
    DenseTensor up;
    up.Resize(out->dims());
    dev_ctx.template Alloc<T>(&gate);
    dev_ctx.template Alloc<T>(&up);
    r = baidu::xpu::api::slice<XPUType>(
        xpu_ctx->x_context(),
        x_data,
        reinterpret_cast<XPUType *>(gate.data<T>()),
        {rows, cols},
        {0, 0},
        {rows, cols / 2});
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::slice failed."));
    r = baidu::xpu::api::slice<XPUType>(
        xpu_ctx->x_context(),
        x_data,
        reinterpret_cast<XPUType *>(up.data<T>()),
        {rows, cols},
        {0, cols / 2},
        {rows, cols});
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::slice failed."));
    r = baidu::xpu::api::gelu<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(gate.data<T>()),
        reinterpret_cast<XPUType *>(gate.data<T>()),
        rows * cols / 2);
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::gelu failed."));
    r = baidu::xpu::api::mul<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(gate.data<T>()),
        reinterpret_cast<const XPUType *>(up.data<T>()),
        reinterpret_cast<XPUType *>(out->data<T>()),
        rows * cols / 2);
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::mul failed."));
  } else if (act_method == "swiglu") {
    DenseTensor gate;
    gate.Resize(out->dims());
    DenseTensor up;
    up.Resize(out->dims());
    dev_ctx.template Alloc<T>(&gate);
    dev_ctx.template Alloc<T>(&up);
    r = baidu::xpu::api::slice<XPUType>(
        xpu_ctx->x_context(),
        x_data,
        reinterpret_cast<XPUType *>(gate.data<T>()),
        {rows, cols},
        {0, 0},
        {rows, cols / 2});
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::slice failed."));
    r = baidu::xpu::api::slice<XPUType>(
        xpu_ctx->x_context(),
        x_data,
        reinterpret_cast<XPUType *>(up.data<T>()),
        {rows, cols},
        {0, cols / 2},
        {rows, cols});
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::slice failed."));
    r = baidu::xpu::api::silu<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(gate.data<T>()),
        reinterpret_cast<XPUType *>(gate.data<T>()),
        rows * cols / 2);
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::silu failed."));
    r = baidu::xpu::api::mul<XPUType>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPUType *>(gate.data<T>()),
        reinterpret_cast<const XPUType *>(up.data<T>()),
        reinterpret_cast<XPUType *>(out->data<T>()),
        rows * cols / 2);
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::mul failed."));
  } else if (act_method == "gelu") {
    r = baidu::xpu::api::gelu<XPUType>(
        xpu_ctx->x_context(),
        x_data,
        reinterpret_cast<XPUType *>(out->data<T>()),
        rows * cols);
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::gelu failed."));
  } else if (act_method == "relu") {
    r = baidu::xpu::api::relu<XPUType>(
        xpu_ctx->x_context(),
        x_data,
        reinterpret_cast<XPUType *>(out->data<T>()),
        rows * cols);
    PADDLE_ENFORCE_EQ(
        r, 0, common::errors::Fatal("baidu::xpu::api::relu failed."));
  } else {
    PD_THROW(
        "NOT supported. "
        "Currently Only Support GeGLU, SwiGLU, GeLU, ReLU");
  }
}

template <typename T, typename Context>
void FusedBiasActKernel(const Context &dev_ctx,
                        const DenseTensor &x,
                        const optional<DenseTensor> &bias,
                        const optional<DenseTensor> &dequant_scales,
                        const optional<DenseTensor> &shift,
                        const optional<DenseTensor> &smooth,
                        const std::string &act_method,
                        const std::string &compute_dtype,
                        float quant_scale,
                        int quant_round_type,
                        float quant_max_bound,
                        float quant_min_bound,
                        DenseTensor *out) {
  if (x.dtype() == DataType::INT32) {
    PADDLE_ENFORCE_NE(
        dequant_scales.get_ptr(),
        nullptr,
        common::errors::InvalidArgument(
            "dequant_scales must be set when Input(x) dtype is INT32."));
    if (quant_scale > 0) {
      dev_ctx.template Alloc<int8_t>(out);
    } else if (compute_dtype == "fp32") {
      dev_ctx.template Alloc<float>(out);
    } else if (compute_dtype == "fp16") {
      dev_ctx.template Alloc<phi::float16>(out);
    } else if (compute_dtype == "bf16") {
      dev_ctx.template Alloc<phi::bfloat16>(out);
    }
    if (out->numel() == 0) return;
    if (compute_dtype == "fp32") {
      return DispatchInt32ComputeImpl<float, Context>(dev_ctx,
                                                      x,
                                                      bias,
                                                      dequant_scales.get(),
                                                      shift,
                                                      smooth,
                                                      act_method,
                                                      quant_scale,
                                                      quant_max_bound,
                                                      out);
    } else if (compute_dtype == "fp16") {
      return DispatchInt32ComputeImpl<phi::float16, Context>(
          dev_ctx,
          x,
          bias,
          dequant_scales.get(),
          shift,
          smooth,
          act_method,
          quant_scale,
          quant_max_bound,
          out);
    } else if (compute_dtype == "bf16") {
      return DispatchInt32ComputeImpl<phi::bfloat16, Context>(
          dev_ctx,
          x,
          bias,
          dequant_scales.get(),
          shift,
          smooth,
          act_method,
          quant_scale,
          quant_max_bound,
          out);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "When Input(x) dtype is INT32, compute_dtype must be fp32, fp16, or "
          "bf16, but got %s.",
          compute_dtype));
    }
  }

  if constexpr (std::is_same_v<T, int32_t>) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Input(x) dtype INT32 requires dequant_scales and a valid "
        "compute_dtype."));
  } else {
    dev_ctx.template Alloc<T>(out);
    if (out->numel() == 0) return;

    return ComputeImpl<T, Context>(dev_ctx, x, bias, act_method, out);
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_bias_act,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedBiasActKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int32_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
