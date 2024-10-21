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

#include "paddle/phi/kernels/random_routing_kernel.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
namespace phi {

template <typename T, typename Context>
void RandomRoutingKernel(const Context& dev_ctx,
                         const DenseTensor& prob_in,
                         const DenseTensor& topk_value_in,
                         const DenseTensor& topk_idx_in,
                         DenseTensor* out) {
  auto topk_idx = &topk_idx_in;
  auto topk_value = &topk_value_in;
  auto prob = &prob_in;

  size_t D = topk_idx->dims()[1];

  phi::DenseTensor topk_value_cpu, prob_cpu;
  phi::Copy(dev_ctx, *topk_value, phi::CPUPlace(), true, &topk_value_cpu);
  phi::Copy(dev_ctx, *prob, phi::CPUPlace(), true, &prob_cpu);
  auto* topk_value_data = topk_value_cpu.data<T>();
  auto* prob_data = prob_cpu.data<T>();
  std::vector<int64_t> out_data(topk_idx->numel());

  for (int64_t idx = 0; idx < topk_idx->numel(); ++idx) {
    size_t row = idx / D;
    size_t col = idx % D;
    if (col == 1 && static_cast<T>(2) * topk_value_data[idx] < prob_data[row]) {
      out_data[idx] = static_cast<int64_t>(-1);
    }
  }
  auto out_dims = out->dims();
  phi::TensorFromVector<int64_t>(out_data, dev_ctx, out);
  out->Resize(out_dims);
}

}  // namespace phi

PD_REGISTER_KERNEL(random_routing,
                   Custom,
                   ALL_LAYOUT,
                   phi::RandomRoutingKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#endif
