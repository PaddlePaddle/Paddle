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

#include "paddle/phi/kernels/multiplex_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void MultiplexGradKernel(const Context& ctx,
                         const DenseTensor& ids,
                         const DenseTensor& out_grad,
                         std::vector<DenseTensor*> ins_grad) {
  size_t idx = -1UL;
  for (size_t i = 0; i < ins_grad.size(); i++) {
    if (ins_grad[i]) {
      ctx.template Alloc<T>(ins_grad[i]);
      auto t = phi::EigenVector<T>::Flatten(*ins_grad[i]);
      t.device(*ctx.eigen_device()) = t.constant(static_cast<T>(0));
      idx = i;
    }
  }
  if (idx == -1UL) return;

  auto rows = ins_grad[idx]->dims()[0];
  auto cols = ins_grad[idx]->numel() / rows;
  DenseTensor index_t_cpu;
  phi::Copy(ctx, ids, phi::CPUPlace(), true, &index_t_cpu);
  auto* index = index_t_cpu.data<int32_t>();
  auto stream = ctx.stream();
  for (auto i = 0; i < rows; i++) {
    size_t k = static_cast<size_t>(index[i]);
    if (ins_grad[k]) {
      memory_utils::Copy(ctx.GetPlace(),
                         ins_grad[k]->data<T>() + i * cols,
                         ctx.GetPlace(),
                         out_grad.data<T>() + i * cols,
                         cols * sizeof(T),
                         stream);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(multiplex_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MultiplexGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
