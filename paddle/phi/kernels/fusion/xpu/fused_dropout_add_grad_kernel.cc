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

// XPU fused_dropout_add backward kernel.
//
// Backward math:
//   y_grad = out_grad              (identity — y enters via addition)
//   x_grad = dropout_grad(out_grad, mask, p)
//
// seed_offset[0] holds the integer seed stored by the forward kernel.
// We regenerate the T-typed dropout mask using that seed, then apply
// phi::DropoutGrad to compute x_grad.
template <typename T, typename Context>
void FusedDropoutAddGradKernel(const Context& dev_ctx,
                               const DenseTensor& seed_offset,
                               const DenseTensor& out_grad,
                               const Scalar& p,
                               bool is_test,
                               const std::string& mode,
                               bool fix_seed,
                               DenseTensor* x_grad,
                               DenseTensor* y_grad) {
  using XPUTypeT = typename XPUTypeTrait<T>::Type;

  int64_t numel = out_grad.numel();
  float dropout_prob = p.to<float>();
  bool is_upscale_in_train = (mode == "upscale_in_train");

  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<T>(y_grad);

  const XPUTypeT* out_grad_data =
      reinterpret_cast<const XPUTypeT*>(out_grad.data<T>());
  XPUTypeT* x_grad_data = reinterpret_cast<XPUTypeT*>(x_grad->data<T>());
  XPUTypeT* y_grad_data = reinterpret_cast<XPUTypeT*>(y_grad->data<T>());

  // y_grad = out_grad (gradient of elementwise add w.r.t. y is identity)
  int r = xpu::copy(dev_ctx.x_context(),
                    reinterpret_cast<const int8_t*>(out_grad_data),
                    reinterpret_cast<int8_t*>(y_grad_data),
                    numel * phi::SizeOf(out_grad.dtype()));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

  if (is_test) {
    // Inference mode: no dropout was applied, x_grad = scale * out_grad
    float scale = is_upscale_in_train ? 1.0f : (1.0f - dropout_prob);
    if (scale == 1.0f) {
      r = xpu::copy(dev_ctx.x_context(),
                    reinterpret_cast<const int8_t*>(out_grad_data),
                    reinterpret_cast<int8_t*>(x_grad_data),
                    numel * phi::SizeOf(out_grad.dtype()));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    } else {
      r = xpu::scale(dev_ctx.x_context(),
                     out_grad_data,
                     x_grad_data,
                     numel,
                     false,
                     scale,
                     0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
    return;
  }

  // Training mode: read the stored seed and regenerate the dropout mask.
  int64_t seed_host[2] = {0LL, 0LL};
  phi::memory_utils::Copy(phi::CPUPlace(),
                          seed_host,
                          seed_offset.place(),
                          seed_offset.data<int64_t>(),
                          sizeof(int64_t) * 2);
  int seed_data = static_cast<int>(seed_host[0]);

  phi::XPUDropoutParam dropout_param;
  dropout_param.dropout_prob = dropout_prob;
  dropout_param.is_upscale_in_train = is_upscale_in_train;
  dropout_param.is_test = false;
  dropout_param.fix_seed = true;  // deterministic replay with stored seed
  dropout_param.tensor_seed = nullptr;
  dropout_param.seed_val = seed_data;

  xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());

  // Regenerate mask: run phi::Dropout on a ones tensor with the same seed.
  // phi::Dropout writes the T-typed mask into mask_buf.
  XPUTypeT* ones_buf = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(numel);
  XPUTypeT* dummy_out = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(numel);
  XPUTypeT* mask_buf = RAII_GUARD.alloc_l3_or_gm<XPUTypeT>(numel);

  r = xpu::constant(dev_ctx.x_context(), ones_buf, numel, XPUTypeT(1));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

  phi::Dropout<XPUTypeT>(
      dev_ctx.x_context(), ones_buf, mask_buf, dummy_out, dropout_param, numel);

  // x_grad = dropout_grad(out_grad, mask, p)
  phi::DropoutGrad<XPUTypeT>(dev_ctx.x_context(),
                             out_grad_data,
                             mask_buf,
                             x_grad_data,
                             dropout_param,
                             numel);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_dropout_add_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDropoutAddGradKernel,
                   float,
                   phi::float16) {}
