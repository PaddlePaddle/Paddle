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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SyncCalcStreamKernel(const Context &dev_ctx,
                          const DenseTensor &x,
                          DenseTensor *out) {
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && !defined(_WIN32)
  phi::backends::gpu::GpuStreamSync(dev_ctx.stream());
#elif defined(PADDLE_WITH_XPU_BKCL)
  auto place = dev_ctx.GetPlace();
  PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::XPU,
                    true,
                    common::errors::PreconditionNotMet(
                        "Sync stream op can run on xpu place only for now."));
  dev_ctx.Wait();
#else
  PADDLE_THROW(common::errors::PreconditionNotMet(
      "PaddlePaddle should compile with GPU."));
#endif
}
}  // namespace phi
