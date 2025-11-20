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

#include "paddle/phi/kernels/quantize_linear_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

// Note: We should re-design this kernel's args when we abandon fluid op
// definition
template <typename T, typename Context>
void DeQuantizeLinearKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const paddle::optional<DenseTensor>& in_scale,
                            const DenseTensor& zero_point,
                            const paddle::optional<DenseTensor>& in_accum,
                            const paddle::optional<DenseTensor>& in_state,
                            int quant_axis,
                            int bit_length,
                            int qmin,
                            int qmax,
                            int round_type,
                            bool is_test,
                            bool only_observer,
                            DenseTensor* out,
                            DenseTensor* out_state,
                            DenseTensor* out_accum,
                            DenseTensor* out_scale) {
  PADDLE_ENFORCE_NE(in_scale.get_ptr(),
                    nullptr,
                    common::errors::PreconditionNotMet(
                        "in_scale can't be nullptr in DeQuantizeLinearKernel"));

  const T* x_data = x.data<T>();
  const T* scale_data = in_scale.get_ptr()->data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);

  if (quant_axis == -1) {
    // step1: out = x * scale
    // int broadcast_mul(Context* xpu_ctx, const T* x, const T* y, T* z, const
    // std::vector<int64_t>& xshape, const std::vector<int64_t>& yshape);
    auto x_dims = x.dims();
    std::vector<int64_t> xshape = common::vectorize<int64_t>(x_dims);
    int r = xpu::broadcast_mul(
        dev_ctx.x_context(), x_data, scale_data, out_data, xshape, {1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");

    // step2: alloc qmax_as_float_xpu
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    float qmax_as_float = qmax;
    float* qmax_as_float_xpu = RAII_GUARD.alloc_l3_or_gm<float>(1);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       static_cast<void*>(qmax_as_float_xpu),
                       CPUPlace(),
                       static_cast<void*>(&qmax_as_float),
                       sizeof(float));

    // step3: out = out / qmax_as_float_xpu
    // int broadcast_div(Context* xpu_ctx, const T* x, const T* y, T* z, const
    // std::vector<int64_t>& xshape, const std::vector<int64_t>& yshape);
    r = xpu::broadcast_div(dev_ctx.x_context(),
                           out_data,
                           qmax_as_float_xpu,
                           out_data,
                           xshape,
                           {1});
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "broadcast_mul");
  } else if (quant_axis == 0) {
    auto x_dims = x.dims();
    const int64_t channel = x_dims[quant_axis];
    const int64_t channel_size = x.numel() / channel;
    // int paddle_clip_dequant_channel(Context* xpu_ctx, const T* x, const T*
    // scale, T* y, int qmax, int64_t channel, int64_t channel_size);
    int r = xpu::paddle_clip_dequant_channel<T>(dev_ctx.x_context(),
                                                x_data,
                                                scale_data,
                                                out_data,
                                                qmax,
                                                channel,
                                                channel_size);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_clip_dequant_channel");
  } else if (quant_axis == 1) {
    // 准备将0和1两个维度对调
    auto x_dims = x.dims();
    std::vector<int64_t> xshape = common::vectorize<int64_t>(x_dims);
    std::vector<int64_t> xshape_back = common::vectorize<int64_t>(x_dims);
    xshape_back[0] = xshape[1];
    xshape_back[1] = xshape[0];
    std::vector<int64_t> trans_axes = {1, 0};
    for (int i = quant_axis + 1; i < x_dims.size(); i++) {
      trans_axes.emplace_back(i);
    }

    // 缓存中间结果
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    T* buffer = RAII_GUARD.alloc_l3_or_gm<T>(x.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(buffer);

    // int transpose(Context* xpu_ctx, const T* x, T* y, const
    // std::vector<int64_t>& xshape,    const std::vector<int64_t>& permute);
    int r = xpu::transpose<T>(
        dev_ctx.x_context(), x_data, buffer, xshape, trans_axes);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

    // 按照axis=0时候的情况进行计算
    const int64_t channel = x_dims[quant_axis];
    const int64_t channel_size = x.numel() / channel;
    // int paddle_clip_dequant_channel(Context* xpu_ctx, const T* x, const T*
    // scale, T* y, int qmax, int64_t channel, int64_t channel_size);
    r = xpu::paddle_clip_dequant_channel<T>(dev_ctx.x_context(),
                                            buffer,
                                            scale_data,
                                            buffer,
                                            qmax,
                                            channel,
                                            channel_size);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_clip_dequant_channel");

    // 算完了再转回去
    r = xpu::transpose<T>(
        dev_ctx.x_context(), buffer, out_data, xshape_back, trans_axes);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "quant axis other than -1, 0, 1 is not supported in XPU"));
  }
}

template <typename T, typename Context>
void QuantizeLinearInferKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const paddle::optional<DenseTensor>& scale,
                               const DenseTensor& zero_point,
                               int quant_axis,
                               int bit_length,
                               int qmin,
                               int qmax,
                               int round_type,
                               bool only_observer,
                               DenseTensor* out) {
  PADDLE_ENFORCE_NE(scale.get_ptr(),
                    nullptr,
                    common::errors::PreconditionNotMet(
                        "in_scale can't be nullptr in DeQuantizeLinearKernel"));

  const T* x_data = x.data<T>();
  const T* scale_data = scale.get_ptr()->data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);

  if (quant_axis == -1) {
    // int paddle_clip_quant(Context* xpu_ctx, const T* x, const T* scale, T* y,
    // int qmax, int64_t n);
    int r = xpu::paddle_clip_quant<T>(
        dev_ctx.x_context(), x_data, scale_data, out_data, qmax, x.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_clip_quant");
  } else if (quant_axis == 0) {
    auto x_dims = x.dims();
    const int64_t channel = x_dims[quant_axis];
    const int64_t channel_size = x.numel() / channel;
    // int paddle_clip_quant_channel(Context* xpu_ctx, const T* x, const T*
    // scale, T* y, int qmax, int64_t channel, int64_t channel_size);
    int r = xpu::paddle_clip_quant_channel<T>(dev_ctx.x_context(),
                                              x_data,
                                              scale_data,
                                              out_data,
                                              qmax,
                                              channel,
                                              channel_size);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_clip_quant_channel");
  } else if (quant_axis == 1) {
    // 准备将0和1两个维度对调
    auto x_dims = x.dims();
    std::vector<int64_t> xshape = common::vectorize<int64_t>(x_dims);
    std::vector<int64_t> xshape_back = common::vectorize<int64_t>(x_dims);
    xshape_back[0] = xshape[1];
    xshape_back[1] = xshape[0];
    std::vector<int64_t> trans_axes = {1, 0};
    for (int i = quant_axis + 1; i < x_dims.size(); i++) {
      trans_axes.emplace_back(i);
    }

    // 缓存中间结果
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    T* buffer = RAII_GUARD.alloc_l3_or_gm<T>(x.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(buffer);

    // int transpose(Context* xpu_ctx, const T* x, T* y, const
    // std::vector<int64_t>& xshape,    const std::vector<int64_t>& permute);
    int r = xpu::transpose<T>(
        dev_ctx.x_context(), x_data, buffer, xshape, trans_axes);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

    // 按照axis=0时候的情况进行计算
    const int64_t channel = x_dims[quant_axis];
    const int64_t channel_size = x.numel() / channel;
    // int paddle_clip_quant_channel(Context* xpu_ctx, const T* x, const T*
    // scale, T* y, int qmax, int64_t channel, int64_t channel_size);
    r = xpu::paddle_clip_quant_channel<T>(dev_ctx.x_context(),
                                          buffer,
                                          scale_data,
                                          buffer,
                                          qmax,
                                          channel,
                                          channel_size);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_clip_quant_channel");

    // 算完了再转回去
    r = xpu::transpose<T>(
        dev_ctx.x_context(), buffer, out_data, xshape_back, trans_axes);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "quant axis other than -1, 0, 1 is not supported in XPU"));
  }
}

// Note: We should re-design this kernel's args when we abandon fluid op
// definition
template <typename T, typename Context>
void QuantizeLinearKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const paddle::optional<DenseTensor>& scale,
                          const DenseTensor& zero_point,
                          const paddle::optional<DenseTensor>& in_accum,
                          const paddle::optional<DenseTensor>& in_state,
                          int quant_axis,
                          int bit_length,
                          int qmin,
                          int qmax,
                          int round_type,
                          bool is_test,
                          bool only_observer,
                          DenseTensor* out,
                          DenseTensor* out_state,
                          DenseTensor* out_accum,
                          DenseTensor* out_scale) {
  if (!is_test) {
    PADDLE_THROW(
        common::errors::Unimplemented("!is_test is not supported in XPU"));
  } else {
    QuantizeLinearInferKernel<T, Context>(dev_ctx,
                                          x,
                                          scale,
                                          zero_point,
                                          quant_axis,
                                          bit_length,
                                          qmin,
                                          qmax,
                                          round_type,
                                          only_observer,
                                          out);
  }
}

template <typename T, typename Context>
void QuantizeLinearDeprecatedInferKernel(const Context& dev_ctx,
                                         const DenseTensor& x,
                                         const DenseTensor& in_scale,
                                         const DenseTensor& zero_point,
                                         int quant_axis,
                                         int bit_length,
                                         int qmin,
                                         int qmax,
                                         int round_type,
                                         bool only_observer,
                                         DenseTensor* out) {
  paddle::optional<phi::DenseTensor> scale =
      paddle::make_optional<phi::DenseTensor>(in_scale);
  QuantizeLinearInferKernel<T, Context>(dev_ctx,
                                        x,
                                        scale,
                                        zero_point,
                                        quant_axis,
                                        bit_length,
                                        qmin,
                                        qmax,
                                        round_type,
                                        only_observer,
                                        out);
}

template <typename T, typename Context>
void DeQuantizeLinearDeprecatedKernel(const Context& dev_ctx,
                                      const DenseTensor& x,
                                      const DenseTensor& in_scale,
                                      const DenseTensor& zero_point,
                                      int quant_axis,
                                      int bit_length,
                                      int qmin,
                                      int qmax,
                                      int round_type,
                                      bool only_observer,
                                      DenseTensor* out) {
  paddle::optional<phi::DenseTensor> scale =
      paddle::make_optional<phi::DenseTensor>(in_scale);
  DeQuantizeLinearKernel<T, Context>(dev_ctx,
                                     x,
                                     scale,
                                     zero_point,
                                     nullptr,
                                     nullptr,
                                     quant_axis,
                                     bit_length,
                                     qmin,
                                     qmax,
                                     round_type,
                                     true,
                                     only_observer,
                                     out,
                                     nullptr,
                                     nullptr,
                                     nullptr);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    quantize_linear, XPU, ALL_LAYOUT, phi::QuantizeLinearKernel, float) {}

PD_REGISTER_KERNEL(
    dequantize_linear, XPU, ALL_LAYOUT, phi::DeQuantizeLinearKernel, float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_KERNEL(quantize_linear_deprecated_infer,
                   XPU,
                   ALL_LAYOUT,
                   phi::QuantizeLinearDeprecatedInferKernel,
                   float) {}

PD_REGISTER_KERNEL(dequantize_linear_deprecated,
                   XPU,
                   ALL_LAYOUT,
                   phi::DeQuantizeLinearDeprecatedKernel,
                   float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
