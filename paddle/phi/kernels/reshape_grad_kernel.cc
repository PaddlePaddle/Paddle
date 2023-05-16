//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/reshape_grad_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#endif
#include <glog/logging.h>
#include "gflags/gflags.h"
DECLARE_string(throw_strided_error_op);

namespace phi {

template <typename Context>
void ReshapeGradKernel(const Context& dev_ctx,
                       const DenseTensor& out_grad,
                       DenseTensor* x_grad) {
  DenseTensor& xx = const_cast<DenseTensor&>(out_grad);
  if (!xx.IsSharedBufferWith(*x_grad)) {
    if (xx.can_not_uses != x_grad->can_not_uses) {
      x_grad->can_not_uses = xx.can_not_uses;
      if (*x_grad->canNotUse == false) {
        *x_grad->canNotUse = *xx.canNotUse;
      }
      xx.can_not_uses->insert(xx.canNotUse);

      xx.can_not_uses->insert(x_grad->canNotUse);
      VLOG(1) << "stride api call log: ReshapeGradKernel";

      if (FLAGS_throw_strided_error_op == "ReshapeGradKernel") {
        PADDLE_THROW(phi::errors::PermissionDenied("wanghuan"));
      }
    }
  }
  auto x_dims = x_grad->dims();
  phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
  x_grad->Resize(x_dims);
}

#ifdef PADDLE_WITH_XPU
template <>
void ReshapeGradKernel<phi::XPUContext>(const XPUContext& dev_ctx,
                                        const DenseTensor& out_grad,
                                        DenseTensor* x_grad) {
  auto x_dims = x_grad->dims();
  dev_ctx.Alloc(x_grad, out_grad.dtype());
  auto* src_ptr = out_grad.data();
  auto* dst_ptr = x_grad->data();
  auto size = out_grad.numel() * phi::SizeOf(out_grad.dtype());
  int ret = xpu::copy(dev_ctx.x_context(),
                      reinterpret_cast<const int8_t*>(src_ptr),
                      reinterpret_cast<int8_t*>(dst_ptr),
                      size);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "copy");
  x_grad->Resize(x_dims);
}
#endif

template <typename Context>
void ReshapeDoubleGradKernel(const Context& dev_ctx,
                             const DenseTensor& out_grad,
                             const DenseTensor& x_grad_grad,
                             DenseTensor* out_grad_grad) {
  ReshapeGradKernel(dev_ctx, x_grad_grad, out_grad_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(reshape_grad,
                                         ALL_LAYOUT,
                                         phi::ReshapeGradKernel) {}
PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(reshape_double_grad,
                                         ALL_LAYOUT,
                                         phi::ReshapeDoubleGradKernel) {}
