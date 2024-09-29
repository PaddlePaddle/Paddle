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
void LimitByCapacityKernel(const Context& dev_ctx,
                           const DenseTensor& expert_count_in,
                           const DenseTensor& capacity_in,
                           int n_worker,
                           DenseTensor* out) {
  auto expert_count = &expert_count_in;
  auto capacity = &capacity_in;

  auto n_expert = expert_count->numel() / n_worker;

  dev_ctx.template Alloc<T>(out);
  std::vector<T> out_data(out->numel());
  phi::DenseTensor expert_count_cpu, capacity_cpu;
  phi::Copy(dev_ctx, *expert_count, phi::CPUPlace(), true, &expert_count_cpu);
  phi::Copy(dev_ctx, *capacity, phi::CPUPlace(), true, &capacity_cpu);

  auto* ec_data = expert_count_cpu.data<T>();
  auto* capacity_data = capacity_cpu.data<T>();
  int eid, wid;
  for (int64_t i = 0; i < expert_count->numel(); ++i) {
    wid = i / n_expert;
    eid = i % n_expert;
    auto proposal = ec_data[i];
    auto cap_left = capacity_data[eid];
    capacity_data[eid] -= proposal;
    if (cap_left >= proposal) {
      out_data[wid * n_expert + eid] = proposal;
    } else if (cap_left >= 0) {
      out_data[wid * n_expert + eid] = cap_left;
    } else {
      out_data[wid * n_expert + eid] = 0;
    }
  }

  auto out_dims = out->dims();
  phi::TensorFromVector<T>(out_data, dev_ctx, out);
  out->Resize(out_dims);
}

}  // namespace phi

PD_REGISTER_KERNEL(limit_by_capacity,
                   Custom,
                   ALL_LAYOUT,
                   phi::LimitByCapacityKernel,
                   int64_t) {}
#endif
