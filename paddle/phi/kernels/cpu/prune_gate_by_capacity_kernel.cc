// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/prune_gate_by_capacity_kernel.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void PruneGateByCapacityKernel(const Context& dev_ctx,
                               const DenseTensor& gate_idx,
                               const DenseTensor& expert_count,
                               int64_t n_expert,
                               int64_t n_worker,
                               DenseTensor* new_gate_idx) {
  PADDLE_THROW(common::errors::Unimplemented(
      "prune_gate_by_capacity is not supported on CPU."));
}

}  // namespace phi

PD_REGISTER_KERNEL(prune_gate_by_capacity,
                   CPU,
                   ALL_LAYOUT,
                   phi::PruneGateByCapacityKernel,
                   int,
                   int64_t) {}
