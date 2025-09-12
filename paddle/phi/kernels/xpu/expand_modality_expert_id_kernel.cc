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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ExpandModalityExpertIDKernel(const Context& dev_ctx,
                                  const DenseTensor& expert_id,
                                  int64_t num_expert_per_modality,
                                  int64_t group_size,
                                  int64_t modality_offset,
                                  bool is_group_expert,
                                  DenseTensor* expert_id_out) {
  dev_ctx.template Alloc<T>(expert_id_out);
  auto expert_id_shape = expert_id.dims();
  int64_t seqlen = expert_id_shape[0];
  int64_t k = expert_id_shape[1];

  int r = xpu::expand_modality_expert_id(dev_ctx.x_context(),
                                         expert_id.data<T>(),
                                         expert_id_out->data<T>(),
                                         seqlen,
                                         k,
                                         num_expert_per_modality,
                                         group_size,
                                         modality_offset,
                                         is_group_expert);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "expand_modality_expert_id");
}
}  // namespace phi

PD_REGISTER_KERNEL(expand_modality_expert_id,
                   XPU,
                   ALL_LAYOUT,
                   phi::ExpandModalityExpertIDKernel,
                   int,
                   int64_t) {}
