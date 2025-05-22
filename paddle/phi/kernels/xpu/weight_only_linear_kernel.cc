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

  if (weight_dtype == "int8") {
    // using TGEMM=int8_wo_t;
    using TGEMM = float;
    baidu::xpu::xblas::FcFusionDesc<TGEMM, float, float> desc{1.0f, 0.0f};
    baidu::xpu::xblas::FcFusionTensor<const int8_t> tensor_w{
        reinterpret_cast<const int8_t*>(weight.data<int8_t>()),
        nullptr,
        n,
        k,
        k,
        true};
    int r1 = baidu::xpu::xblas::fc_fusion<XPUType,
                                          int8_t,
                                          XPUType,
                                          XPUType,
                                          TGEMM,
                                          float,
                                          float,
                                          float,
                                          float>(dev_ctx.x_context(),
                                                 tensor_x,
                                                 tensor_w,
                                                 tensor_y_const,
                                                 tensor_y,
                                                 desc,
                                                 epilogue);
    PD_CHECK(r1 == 0, "xblas::fc_fusion failed");
  } else if (weight_dtype == "int4") {
    // baidu::xpu::xblas::FcFusionDesc<int4_wo_int15, float, XPUType>
    // desc{1.0f, 0.0f};
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
    PD_THROW("unsupported weight_dtype=int4");
  } else {
    PD_THROW("unsupported weight_dtype: ", weight_dtype);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(weight_only_linear,
                   XPU,
                   ALL_LAYOUT,
                   phi::WeightOnlyLinearKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
