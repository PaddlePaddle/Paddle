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

#include "paddle/phi/kernels/lamb_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void LambKernel(const Context& dev_ctx,
                const DenseTensor& param,
                const DenseTensor& grad,
                const DenseTensor& learning_rate,
                const DenseTensor& moment1,
                const DenseTensor& moment2,
                const DenseTensor& beta1_pow,
                const DenseTensor& beta2_pow,
                const paddle::optional<DenseTensor>& master_param,
                const paddle::optional<DenseTensor>& skip_update,
                float weight_decay,
                float beta1,
                float beta2,
                float epsilon,
                bool always_adapt,
                bool multi_precision,
                DenseTensor* param_outs,
                DenseTensor* moment1_out,
                DenseTensor* moment2_out,
                DenseTensor* beta1_pow_out,
                DenseTensor* beta2_pow_out,
                DenseTensor* master_param_outs) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  if (!multi_precision) {
    constexpr auto kIsSameType = std::is_same<T, MT>::value;
    PADDLE_ENFORCE_EQ(
        kIsSameType,
        true,
        common::errors::InvalidArgument(
            "When multi_precision=False, T and MT must be the same type."));
  }
  bool cpu_skip_update = false;
  if (skip_update && skip_update->IsInitialized()) {
    if (skip_update->place().GetType() == phi::AllocationType::CPU) {
      cpu_skip_update = *(skip_update->data<bool>());
    } else {
      const bool* skip_update_flag = skip_update->data<bool>();
      memory_utils::Copy(phi::CPUPlace(),
                         static_cast<void*>(&cpu_skip_update),
                         dev_ctx.GetPlace(),
                         static_cast<const void*>(skip_update_flag),
                         sizeof(bool));
    }
  }
  if (cpu_skip_update) {
    return;
  }

  // tensor --> data_ptr
  // inputs
  const XPUType* param_ptr = reinterpret_cast<const XPUType*>(param.data<T>());
  const XPUType* grad_ptr = reinterpret_cast<const XPUType*>(grad.data<T>());
  const MT* learning_rate_ptr = learning_rate.data<MT>();
  const MT* moment1_ptr = moment1.data<MT>();
  const MT* moment2_ptr = moment2.data<MT>();
  const MT* beta1_pow_ptr = beta1_pow.data<MT>();

  const MT* beta2_pow_ptr = beta2_pow.data<MT>();

  const MT* master_param_ptr = nullptr;
  if (multi_precision) {
    master_param_ptr = master_param.get_ptr()->data<MT>();
  }

  // outputs
  XPUType* param_outs_ptr =
      reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(param_outs));
  MT* moment1_out_ptr = dev_ctx.template Alloc<MT>(moment1_out);
  MT* moment2_out_ptr = dev_ctx.template Alloc<MT>(moment2_out);

  MT* master_param_outs_ptr = nullptr;
  if (multi_precision) {
    if (master_param_outs->numel() != master_param.get_ptr()->numel()) {
      master_param_outs->Resize(master_param.get_ptr()->dims());
    }
    master_param_outs_ptr = dev_ctx.template Alloc<MT>(master_param_outs);
  }

  MT* beta1_pow_out_ptr = nullptr;
  MT* beta2_pow_out_ptr = nullptr;
  MT* beta1_pow_xpu_ptr = nullptr;
  MT* beta2_pow_xpu_ptr = nullptr;

  xpu::Context* xpu_ctx = dev_ctx.x_context();
  xpu::ctx_guard RAII_GUARD(xpu_ctx);

  if (beta1_pow.place().GetType() == phi::AllocationType::CPU) {
    beta1_pow_xpu_ptr = RAII_GUARD.alloc_l3_or_gm<MT>(beta1_pow.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(beta1_pow_out_ptr);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       beta1_pow_xpu_ptr,
                       beta1_pow.place(),
                       beta1_pow.data<MT>(),
                       sizeof(MT) * beta1_pow.numel());
    beta1_pow_ptr = beta1_pow_xpu_ptr;
    beta1_pow_out_ptr = RAII_GUARD.alloc_l3_or_gm<MT>(beta1_pow_out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(beta1_pow_out_ptr);

  } else {
    beta1_pow_out_ptr = dev_ctx.template Alloc<MT>(beta1_pow_out);
  }
  if (beta2_pow.place().GetType() == phi::AllocationType::CPU) {
    beta2_pow_xpu_ptr = RAII_GUARD.alloc_l3_or_gm<MT>(beta2_pow.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(beta2_pow_xpu_ptr);
    memory_utils::Copy(dev_ctx.GetPlace(),
                       beta2_pow_xpu_ptr,
                       beta2_pow.place(),
                       beta2_pow.data<MT>(),
                       sizeof(MT) * beta2_pow.numel());
    beta2_pow_ptr = beta2_pow_xpu_ptr;

    beta2_pow_out_ptr = RAII_GUARD.alloc_l3_or_gm<MT>(beta2_pow_out->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(beta2_pow_out_ptr);

  } else {
    beta2_pow_out_ptr = dev_ctx.template Alloc<MT>(beta2_pow_out);
  }

  const MT* param_calc_ptr = nullptr;
  const MT* grad_calc_ptr = nullptr;
  MT* param_outs_calc_ptr = nullptr;

  if (std::is_same<T, phi::dtype::float16>::value) {
    MT* param_float = RAII_GUARD.alloc_l3_or_gm<MT>(param.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(param_float);
    MT* grad_float = RAII_GUARD.alloc_l3_or_gm<MT>(grad.numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(grad_float);
    MT* param_outs_float = RAII_GUARD.alloc_l3_or_gm<MT>(param_outs->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(param_outs_float);
    int r =
        xpu::cast<XPUType, MT>(xpu_ctx, param_ptr, param_float, param.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    r = xpu::cast<XPUType, MT>(xpu_ctx, grad_ptr, grad_float, grad.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    param_calc_ptr = param_float;
    grad_calc_ptr = grad_float;

    param_outs_calc_ptr = param_outs_float;
  } else {
    param_calc_ptr = reinterpret_cast<const MT*>(param_ptr);
    grad_calc_ptr = reinterpret_cast<const MT*>(grad_ptr);
    param_outs_calc_ptr = reinterpret_cast<MT*>(param_outs_ptr);
  }
  int r = xpu::lamb<MT>(
      xpu_ctx,
      grad_calc_ptr,
      moment1_ptr,
      moment2_ptr,
      (multi_precision ? master_param_ptr : param_calc_ptr),
      beta1_pow_ptr,
      beta2_pow_ptr,
      moment1_out_ptr,
      moment2_out_ptr,
      (multi_precision ? master_param_outs_ptr : param_outs_calc_ptr),
      beta1_pow_out_ptr,
      beta2_pow_out_ptr,
      beta1,
      beta2,
      epsilon,
      weight_decay,
      learning_rate_ptr,
      param.numel());

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "lamb");
  if (std::is_same<T, phi::dtype::float16>::value && multi_precision == false) {
    int r = xpu::cast<MT, XPUType>(
        xpu_ctx, param_outs_calc_ptr, param_outs_ptr, param_outs->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  }

  if (beta1_pow.place().GetType() == phi::AllocationType::CPU) {
    // copy beta1_pow_out from xpu to cpu
    memory_utils::Copy(beta1_pow.place(),
                       dev_ctx.template HostAlloc<MT>(beta1_pow_out),
                       dev_ctx.GetPlace(),
                       beta1_pow_out_ptr,
                       sizeof(MT) * beta1_pow_out->numel());
  }
  if (beta2_pow.place().GetType() == phi::AllocationType::CPU) {
    // copy beta2_pow_out from xpu to cpu
    memory_utils::Copy(beta2_pow.place(),
                       dev_ctx.template HostAlloc<MT>(beta2_pow_out),
                       dev_ctx.GetPlace(),
                       beta2_pow_out_ptr,
                       sizeof(MT) * beta2_pow_out->numel());
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    lamb, XPU, ALL_LAYOUT, phi::LambKernel, float, phi::dtype::float16) {
  kernel->InputAt(5).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(6).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(3).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(4).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(5).SetDataType(phi::DataType::UNDEFINED);
}
