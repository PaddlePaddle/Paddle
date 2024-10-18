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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context>
void PruneGateByCapacityKernel(const Context& dev_ctx,
                               const DenseTensor& gate_idx_in,
                               const DenseTensor& expert_count_in,
                               int64_t n_expert,
                               int64_t n_worker,
                               DenseTensor* new_gate_idx) {
  auto* gate_idx = &gate_idx_in;
  auto* expert_count = &expert_count_in;

  dev_ctx.template Alloc<T>(new_gate_idx);

  phi::DenseTensor expert_count_cpu, gate_idx_cpu;
  phi::Copy(dev_ctx, *expert_count, phi::CPUPlace(), true, &expert_count_cpu);
  phi::Copy(dev_ctx, *gate_idx, phi::CPUPlace(), true, &gate_idx_cpu);
  auto expert_count_data = expert_count_cpu.data<T>();
  auto gate_idx_data = gate_idx_cpu.data<T>();
  std::vector<T> new_gate_idx_data(gate_idx->numel());
  for (auto i = 0; i < gate_idx->numel(); ++i) {
    auto orig_cap = expert_count_data[gate_idx_data[i]]--;
    if (orig_cap <= 0) {
      new_gate_idx_data[i] = -1;
    } else {
      new_gate_idx_data[i] = gate_idx_data[i];
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(prune_gate_by_capacity,
                   Custom,
                   ALL_LAYOUT,
                   phi::PruneGateByCapacityKernel,
                   int64_t) {}
#endif
