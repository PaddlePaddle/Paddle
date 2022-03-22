// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/multiplex_kernel.h"

#include "paddle/phi/api/lib/utils/tensor_utils.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MultiplexKernel(const Context& ctx,
                     const std::vector<const DenseTensor*>& ins,
                     const DenseTensor& ids,
                     DenseTensor* out) {
  ctx.template Alloc<T>(out);
  for (size_t i = 0; i < ins.size(); ++i) {
    PADDLE_ENFORCE_GT(
        ins[i]->numel(),
        0,
        errors::OutOfRange(
            "indexing will be out of bounds with size 0 for the %d-th input.",
            i));
  }

  auto rows = ins[0]->dims()[0];
  auto cols = ins[0]->numel() / rows;
  DenseTensor index_t_cpu;
  paddle::framework::TensorCopySync(ids, phi::CPUPlace(), &index_t_cpu);
  auto* index = index_t_cpu.data<int32_t>();
  auto stream = ctx.stream();
  for (auto i = 0; i < rows; i++) {
    int32_t k = index[i];
    PADDLE_ENFORCE_GE(
        k, 0, errors::PreconditionNotMet("index must be nonnegative."));
    PADDLE_ENFORCE_LT(static_cast<size_t>(k),
                      ins.size(),
                      errors::PreconditionNotMet(
                          "index exceeds the number of candidate tensors."));
    paddle::memory::Copy(ctx.GetPlace(),
                         out->data<T>() + i * cols,
                         ctx.GetPlace(),
                         ins[k]->data<T>() + i * cols,
                         cols * sizeof(T),
                         stream);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(multiplex,
                   GPU,
                   ALL_LAYOUT,
                   phi::MultiplexKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
