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
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_fused_common_function.h"

namespace phi {
namespace fusion {

// XPU fused_dropout_add forward kernel.
//
// Forward math (upscale_in_train, training=True):
//   mask        = dropout_mask(seed)        -- T-typed mask via xpu::dropout
//   dropout_out = x * mask
//   out         = dropout_out + y
//
// The resolved integer seed is stored in seed_offset[0] (INT64, shape {2})
// so that the backward kernel can regenerate the same mask deterministically.
template <typename T, typename Context>
void FusedDropoutAddKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           const paddle::optional<DenseTensor>& seed_tensor,
                           const Scalar& p,
                           bool is_test,
                           const std::string& mode,
                           int seed,
                           bool fix_seed,
                           DenseTensor* out,
                           DenseTensor* seed_offset) {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

  int64_t numel = x.numel();
  float dropout_prob = p.to<float>();
  bool is_upscale_in_train = (mode == "upscale_in_train");

  dev_ctx.template Alloc<T>(out);
  // seed_offset stores [seed_data, 0] as int64, matching InferMeta dtype.
  dev_ctx.template Alloc<int64_t>(seed_offset);

  const XPUTypeT* x_data = reinterpret_cast<const XPUTypeT*>(x.data<T>());
  const XPUTypeT* y_data = reinterpret_cast<const XPUTypeT*>(y.data<T>());
  XPUTypeT* out_data = reinterpret_cast<XPUTypeT*>(out->data<T>());

  // Determine the actual integer seed, following dropout_kernel.cc logic.
  int seed_data = 0;
  const DenseTensor* seed_tensor_ptr = seed_tensor.get_ptr();
  if (seed_tensor_ptr != nullptr) {
    if (seed_tensor_ptr->place().GetType() == phi::AllocationType::XPU) {
      phi::memory_utils::Copy(phi::CPUPlace(),
                              &seed_data,
                              seed_tensor_ptr->place(),
                              seed_tensor_ptr->data<int>(),
                              sizeof(int));
    } else {
      seed_data = *(seed_tensor_ptr->data<int>());
    }
  } else {
    seed_data = fix_seed ? seed : 0;
  }
  if (seed_data == 0) {
    seed_data = static_cast<int>(dev_ctx.GetGenerator()->Random64());
  }

  // Store resolved seed in seed_offset so backward can regenerate the mask.
  int64_t seed_host[2] = {static_cast<int64_t>(seed_data), 0LL};
  phi::memory_utils::Copy(seed_offset->place(),
                          seed_offset->data<int64_t>(),
                          phi::CPUPlace(),
                          seed_host,
                          sizeof(int64_t) * 2);

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  if (is_test) {
    // Inference mode: out = scale(x) + y  (no dropout)
    float scale = is_upscale_in_train ? 1.0f : (1.0f - dropout_prob);
    if (scale == 1.0f) {
      int r = xpu::add(dev_ctx.x_context(), x_data, y_data, out_data, numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
    } else {
      XPUTypeT* scaled_x = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(numel);
      int r = xpu::scale(
          dev_ctx.x_context(), x_data, scaled_x, numel, false, scale, 0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
      r = xpu::add(dev_ctx.x_context(), scaled_x, y_data, out_data, numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
    }
    return;
  }

  // Training mode: apply dropout to x with resolved seed, then add y.
  phi::XPUDropoutParam dropout_param;
  dropout_param.dropout_prob = dropout_prob;
  dropout_param.is_upscale_in_train = is_upscale_in_train;
  dropout_param.is_test = false;
  dropout_param.fix_seed = true;  // seed_data already resolved above
  dropout_param.tensor_seed = nullptr;
  dropout_param.seed_val = seed_data;

  // Temporary buffers: T-typed mask (for phi::Dropout) and dropout output.
  XPUTypeT* mask_tmp = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(numel);
  XPUTypeT* dropout_out = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(numel);

  phi::Dropout<XPUTypeT>(
      dev_ctx.x_context(), x_data, mask_tmp, dropout_out, dropout_param, numel);

  int r = xpu::add(dev_ctx.x_context(), dropout_out, y_data, out_data, numel);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_dropout_add,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDropoutAddKernel,
                   float,
                   phi::float16) {}
