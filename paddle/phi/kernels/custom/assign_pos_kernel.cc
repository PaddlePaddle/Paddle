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

#include "paddle/phi/kernels/assign_pos_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context>
void AssignPosKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& cum_count_in,
                     const DenseTensor& eff_num_len_in,
                     DenseTensor* out) {
  // assign pos decides which tokens should be fetched belong to specially
  // counter orderly.
  auto cum_count = &cum_count_in;      // (counter number) int32 | int64
  auto numbers = &x;                   // (batch_size * seq_len, topk) int32
  auto eff_num_len = &eff_num_len_in;  // (sum(cum_count))
  // out: (cum_count) value ranges
  // from 0 to batch_size *
  // seq_len * topk

  phi::DenseTensor cpu_eff_num_len;
  int64_t cpu_eff_num_len_data = 0;
  if (eff_num_len->place().GetType() == phi::AllocationType::CPU) {
    cpu_eff_num_len_data = eff_num_len->data<T>()[0];
  } else {
    phi::Copy(dev_ctx, *eff_num_len, phi::CPUPlace(), true, &cpu_eff_num_len);
    cpu_eff_num_len_data = cpu_eff_num_len.data<T>()[0];
  }

  out->Resize({cpu_eff_num_len_data});
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor numbers_cpu, cum_count_cpu;
  phi::Copy(dev_ctx, *numbers, phi::CPUPlace(), true, &numbers_cpu);
  phi::Copy(dev_ctx, *cum_count, phi::CPUPlace(), true, &cum_count_cpu);
  auto* numbers_data = numbers_cpu.data<T>();
  auto* cum_count_data = cum_count_cpu.data<T>();

  std::vector<T> out_data(cpu_eff_num_len_data);
  for (int64_t i = 0; i < numbers->numel(); ++i) {
    int number_idx = numbers_data[i];
    if (number_idx > -1) {
      cum_count_data[number_idx] -= 1;
      int p = cum_count_data[number_idx];
      out_data[p] = i;
    }
  }
  phi::TensorFromVector<int64_t>(out_data, dev_ctx, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    assign_pos, Custom, ALL_LAYOUT, phi::AssignPosKernel, int64_t) {}
#endif
