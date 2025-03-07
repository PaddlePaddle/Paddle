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

#include <xft/xdnn_plugin.h>
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#ifdef PADDLE_WITH_XPU_XRE5
#include "xblas/xblas_legacy_api.h"
#endif

namespace phi {
template <typename T, typename Context>
void WeightOnlyLinearKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& weight,
                            const paddle::optional<DenseTensor>& bias,
                            const DenseTensor& weight_scale,
                            const std::string& weight_dtype,
                            const int32_t arch,
                            const int32_t group_size,
                            DenseTensor* out) {
  if (dev_ctx.x_context()->dev().type() == baidu::xpu::api::kXPU3) {
    using XPUType = typename XPUTypeTrait<T>::Type;
    int64_t n = weight.dims()[0];
    int64_t k = weight.dims()[1];
    int64_t m = x.numel() / k;
    if (weight_dtype == "int4") {
      n = n * 2;
    }
    out->Resize({static_cast<int64_t>(m), static_cast<int64_t>(n)});
    dev_ctx.template Alloc<T>(out);

    DenseTensor bias_fp32;
    if (bias.is_initialized() && bias.get().dtype() == phi::DataType::FLOAT16) {
      bias_fp32.Resize(bias.get().dims());
      dev_ctx.template Alloc<float>(&bias_fp32);
      int r = baidu::xpu::api::cast<XPUType, float>(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(
              bias.get().data<phi::dtype::float16>()),
          bias_fp32.data<float>(),
          n);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    }
    auto input_x = reinterpret_cast<const XPUType*>(x.data<T>());
    auto input_y = reinterpret_cast<XPUType*>(out->data<T>());

    baidu::xpu::xblas::FcFusionTensor<const XPUType> tensor_x{
        input_x, nullptr, m, k, k, false};
    baidu::xpu::xblas::FcFusionTensor<const XPUType> tensor_y_const{
        input_y, nullptr, m, n, n, false};
    baidu::xpu::xblas::FcFusionTensor<XPUType> tensor_y{
        input_y, nullptr, m, n, n, false};

    baidu::xpu::xblas::FcFusionEpilogue<float, float> epilogue{
        api::Activation_t::LINEAR,
        bias.is_initialized() ? (bias.get().dtype() == phi::DataType::FLOAT16
                                     ? bias_fp32.data<float>()
                                     : bias.get().data<float>())
                              : nullptr,
        nullptr,
        weight_scale.dims().size() != 0 ? weight_scale.data<float>() : nullptr,
        0,
        1,
        nullptr};
    if (weight_dtype == "int4") {
      // baidu::xpu::xblas::FcFusionDesc<int4_wo_int15, float, XPUType>
      // desc{1.0f,
      //                                                                     0.0f};
      // baidu::xpu::xblas::FcFusionTensor<const int4_t> tensor_w{
      //     reinterpret_cast<const int4_t*>(weight.data<int8_t>()),
      //     nullptr,
      //     n,
      //     k,
      //     k,
      //     true};
      // int r1 = baidu::xpu::xblas::fc_fusion<XPUType,
      //                                       int4_t,
      //                                       XPUType,
      //                                       XPUType,
      //                                       int4_wo_int15,  // int8_wo_t
      //                                       float,
      //                                       XPUType,
      //                                       float,
      //                                       float>(dev_ctx.x_context(),
      //                                              tensor_x,
      //                                              tensor_w,
      //                                              tensor_y_const,
      //                                              tensor_y,
      //                                              desc,
      //                                              epilogue);
      // PD_CHECK(r1 == 0, "xblas::fc_fusion failed");

    } else {
      baidu::xpu::xblas::FcFusionDesc<float, float, float> desc{1.0f, 0.0f};
      baidu::xpu::xblas::FcFusionTensor<const XPUType> tensor_w{
          // reinterpret_cast<const
          // XPUType*>(weight.data<phi::dtype::bfloat16>()), nullptr, k, n, n,
          // true};
          reinterpret_cast<const XPUType*>(weight.data<T>()),
          nullptr,
          n,
          k,
          k,
          true};
      int r1 = baidu::xpu::xblas::fc_fusion<XPUType,
                                            XPUType,
                                            XPUType,
                                            XPUType,
                                            float,
                                            float,
                                            float,
                                            float,
                                            float,
                                            XPUType>(dev_ctx.x_context(),
                                                     tensor_x,
                                                     tensor_w,
                                                     tensor_y_const,
                                                     tensor_y,
                                                     desc,
                                                     epilogue);
      PD_CHECK(r1 == 0, "xblas::fc_fusion failed");
    }
  } else {
    PADDLE_ENFORCE_EQ(
        weight_dtype,
        "int8",
        common::errors::Fatal(
            "WeightOnlyLinearKernel xpu just support int8 weight only"));
    phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
    auto xpu_ctx = static_cast<const phi::XPUContext*>(&dev_ctx);
    dev_ctx.template Alloc<T>(out);
    int r = 0;
    switch (x.dtype()) {
      case phi::DataType::FLOAT16: {
        using XPUType = typename XPUTypeTrait<phi::dtype::float16>::Type;
        int n = weight.dims()[0];
        int k = weight.dims()[1];
        int m = x.numel() / k;
        DenseTensor max_value;
        max_value.Resize(weight_scale.dims());
        dev_ctx.template Alloc<float>(&max_value);
        if (weight_scale.dtype() == phi::DataType::FLOAT16) {
          DenseTensor max_value_fp16;
          max_value_fp16.Resize(weight_scale.dims());
          dev_ctx.template Alloc<phi::dtype::float16>(&max_value_fp16);
          r = baidu::xpu::api::scale(
              xpu_ctx->x_context(),
              reinterpret_cast<const XPUType*>(
                  weight_scale.data<phi::dtype::float16>()),
              reinterpret_cast<XPUType*>(
                  max_value_fp16.data<phi::dtype::float16>()),
              weight_scale.numel(),
              false,
              weight_dtype == "int8" ? 127.f : 7.f,
              0.f);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
          r = baidu::xpu::api::cast<XPUType, float>(
              xpu_ctx->x_context(),
              reinterpret_cast<const XPUType*>(
                  max_value_fp16.data<phi::dtype::float16>()),
              max_value.data<float>(),
              max_value.numel());
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
        } else if (weight_scale.dtype() == phi::DataType::FLOAT32) {
          r = baidu::xpu::api::scale(xpu_ctx->x_context(),
                                     weight_scale.data<float>(),
                                     max_value.data<float>(),
                                     weight_scale.numel(),
                                     false,
                                     weight_dtype == "int8" ? 127.f : 7.f,
                                     0.f);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "Only support that weight scale as type float32 ot float16."));
        }

        DenseTensor bias_fp32;
        if (bias.is_initialized() &&
            bias.get().dtype() == phi::DataType::FLOAT16) {
          bias_fp32.Resize(bias.get().dims());
          dev_ctx.template Alloc<float>(&bias_fp32);
          r = baidu::xpu::api::cast<XPUType, float>(
              xpu_ctx->x_context(),
              reinterpret_cast<const XPUType*>(
                  bias.get().data<phi::dtype::float16>()),
              bias_fp32.data<float>(),
              n);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
        }
        if (weight_dtype == "int8") {
          r = baidu::xpu::api::
              gpt_fc_fusion<XPUType, int8_t, XPUType, int8_wo_t>(
                  xpu_ctx->x_context(),
                  reinterpret_cast<const XPUType*>(
                      x.data<phi::dtype::float16>()),
                  weight.data<int8_t>(),
                  reinterpret_cast<XPUType*>(out->data<phi::dtype::float16>()),
                  m,
                  n,
                  k,
                  false,
                  true,
                  nullptr,
                  nullptr,
                  nullptr,
                  k,
                  k,
                  n,
                  1.0f,
                  0.0f,
                  bias.is_initialized()
                      ? (bias.get().dtype() == phi::DataType::FLOAT16
                             ? bias_fp32.data<float>()
                             : bias.get().data<float>())
                      : nullptr,
                  baidu::xpu::api::Activation_t::LINEAR,
                  max_value.data<float>());
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "gpt_fc_fusion");
        } else if (weight_dtype == "int4") {
          PD_THROW("only support int8 weight only now");
        }
        return;
      }
      default: {
        PD_THROW("unsupported data type");
      }
    }
  }
}
}  // namespace phi
PD_REGISTER_KERNEL(weight_only_linear,
                   XPU,
                   ALL_LAYOUT,
                   phi::WeightOnlyLinearKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
