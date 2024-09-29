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

#pragma once

#include <string>

#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#elif defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#endif

namespace phi {

#if defined(PADDLE_WITH_XPU_BKCL)
static void XPUStreamSync(XPUStream stream) {
  PADDLE_ENFORCE_XDNN_SUCCESS(xpu_wait(stream), "xpu_wait");
}
#endif

template <typename T, typename Context>
void SyncCommStreamKernel(const Context &dev_ctx,
                          const std::vector<const DenseTensor *> &x UNUSED,
                          int ring_id UNUSED,
                          std::vector<DenseTensor *> out UNUSED) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  phi::backends::gpu::GpuStreamSync(dev_ctx.stream());
#elif defined(PADDLE_WITH_XPU_BKCL)
  XPUStreamSync(dev_ctx.stream());
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle should compile with GPU or XPU."));
#endif
}

}  // namespace phi
