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

#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

namespace phi {

#if defined(PADDLE_WITH_XPU_BKCL)
static void XPUStreamSync(XPUStream stream) {
  PADDLE_ENFORCE_XDNN_SUCCESS(xpu_wait(stream), "xpu_wait");
}
#endif

template <typename T, typename Context>
void BarrierKernel(const Context &dev_ctx,
                   const DenseTensor &x,
                   DenseTensor *out) {
#if defined(PADDLE_WITH_XPU_BKCL)
  auto in = &x;
  auto comm_ctx = static_cast<phi::distributed::BKCLCommContext *>(
      dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(comm_ctx,
                    nullptr,
                    common::errors::Unavailable(
                        "BKCLCommContext is nullptr, collective op should "
                        "has ring_id attr."));
  XPUStream stream = dev_ctx.stream();
  BKCLOp bkcl_reduce_type = BKCL_ADD;
  comm_ctx->AllReduce(out, *in, bkcl_reduce_type, stream);
  XPUStreamSync(stream);
#else
  PADDLE_THROW(
      common::errors::Unavailable("PaddlePaddle should compile with BKCL."));
#endif
}
}  // namespace phi

PD_REGISTER_KERNEL(barrier, XPU, ALL_LAYOUT, phi::BarrierKernel, int) {}
