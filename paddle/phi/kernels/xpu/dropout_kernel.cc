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

#include "paddle/phi/kernels/dropout_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DropoutRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& seed_tensor,
                      const Scalar& p,
                      bool is_test,
                      const std::string& mode,
                      int seed,
                      bool fix_seed,
                      DenseTensor* out,
                      DenseTensor* mask) {
  bool is_upscale = (mode == "upscale_in_train");
  dev_ctx.template Alloc<T>(out);
  if (mask) {
    dev_ctx.template Alloc<uint8_t>(mask);
  }

  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto* x_data = x.data<T>();
  auto* y_data = out->data<T>();
  float dropout_prob = p.to<float>();

  if (!is_test && mask) {
    int seed_data = 0;
    if (seed_tensor.get_ptr() != nullptr) {
      if ((seed_tensor->place()).GetType() == phi::AllocationType::XPU) {
        memory_utils::Copy(phi::CPUPlace(),
                           &seed_data,
                           seed_tensor->place(),
                           seed_tensor->data<int>(),
                           sizeof(int));
      } else {
        seed_data = *(seed_tensor->data<int>());
      }
    } else {
      seed_data = fix_seed ? seed : 0;
    }
    if (seed_data == 0) {
      seed_data = dev_ctx.GetGenerator()->Random64();
    }

    auto* mask_data = mask->data<uint8_t>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    auto dev_version =
        phi::backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId());
    // Special case when dropout_prob is 1.0
    if (dropout_prob == 1.0f) {
      int r = xpu::constant(dev_ctx.x_context(),
                            reinterpret_cast<XPUType*>(y_data),
                            out->numel(),
                            XPUType(0));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
      r = xpu::constant(
          dev_ctx.x_context(), mask_data, mask->numel(), uint8_t(0));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
      return;
    }
    if (dev_version == phi::backends::xpu::XPUVersion::XPU3) {
      // int dropout_v3(Context* ctx, const T* input, T* res, uint8_t* mask,
      // unsigned int seed, int64_t n, bool is_upscale, float dropout_prob);
      int r = xpu::dropout_v3(dev_ctx.x_context(),
                              reinterpret_cast<const XPUType*>(x_data),
                              reinterpret_cast<XPUType*>(y_data),
                              mask_data,
                              seed_data,
                              mask->numel(),
                              is_upscale,
                              dropout_prob);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout_v3");
    } else {
      XPUType* mask_tmp_data =
          RAII_GUARD.alloc_l3_or_gm<XPUType>(mask->numel());
      // int dropout(Context* ctx, const T* input, T* res, T* mask, unsigned int
      // seed, int64_t n, bool is_upscale, float dropout_prob);
      int r = xpu::dropout(dev_ctx.x_context(),
                           reinterpret_cast<const XPUType*>(x_data),
                           reinterpret_cast<XPUType*>(y_data),
                           mask_tmp_data,
                           seed_data,
                           mask->numel(),
                           is_upscale,
                           dropout_prob);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "dropout");
      r = xpu::cast<XPUType, uint8_t>(
          dev_ctx.x_context(), mask_tmp_data, mask_data, mask->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    }
  } else {
    if (is_upscale) {
      // y = x
      int ret = xpu::copy(dev_ctx.x_context(),
                          reinterpret_cast<const int8_t*>(x_data),
                          reinterpret_cast<int8_t*>(y_data),
                          x.numel() * phi::SizeOf(x.dtype()));
      PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy");
    } else {
      int r = xpu::scale(dev_ctx.x_context(),
                         reinterpret_cast<const XPUType*>(x_data),
                         reinterpret_cast<XPUType*>(y_data),
                         x.numel(),
                         false,
                         1.0f - dropout_prob,
                         0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(dropout,
                   XPU,
                   ALL_LAYOUT,
                   phi::DropoutRawKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}
