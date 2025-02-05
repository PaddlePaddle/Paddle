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

#include "paddle/phi/kernels/warpctc_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void WarpctcGradKernel(const Context& dev_ctx,
                       const DenseTensor& logits,
                       const paddle::optional<DenseTensor>& logits_length,
                       const DenseTensor& warpctcgrad,
                       const DenseTensor& loss_grad,
                       int blank,
                       bool norm_by_times,
                       DenseTensor* logits_grad) {
  dev_ctx.template Alloc<T>(logits_grad);

  bool has_logits_length = logits_length.is_initialized();
  if (!has_logits_length) {
    PADDLE_THROW(common::errors::External(
        "XPU only support logits_length is_initialized"));
  }

  int max_seq_length = warpctcgrad.dims()[0];  // Tmax
  int num_sequences = warpctcgrad.dims()[1];   // B
  int seq_width = warpctcgrad.dims()[2];       // D
  auto* logits_length_ptr = logits_length.get_ptr();

  int r = xpu::ctc_loss_grad<T, int64_t>(dev_ctx.x_context(),
                                         loss_grad.data<T>(),
                                         logits_grad->data<T>(),
                                         warpctcgrad.data<T>(),
                                         max_seq_length,
                                         num_sequences,
                                         seq_width,
                                         logits_length_ptr->data<int64_t>(),
                                         norm_by_times);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "ctc_loss_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    warpctc_grad, XPU, ALL_LAYOUT, phi::WarpctcGradKernel, float) {}
