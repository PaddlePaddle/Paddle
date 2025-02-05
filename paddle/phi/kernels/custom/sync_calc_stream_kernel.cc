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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {
template <typename T, typename Context>
void SyncCalcStreamKernel(const Context &dev_ctx,
                          const DenseTensor &x,
                          DenseTensor *out) {
  dev_ctx.GetStream()->Synchronize();
}
}  // namespace phi

PD_REGISTER_KERNEL(sync_calc_stream,
                   Custom,
                   ALL_LAYOUT,
                   phi::SyncCalcStreamKernel,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
