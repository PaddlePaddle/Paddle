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

#include <memory>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/shuffle_batch_kernel.h"

namespace phi {

template <typename T, typename Context>
void ShuffleBatchGradKernel(const Context& dev_ctx,
                            const DenseTensor& shuffleidx,
                            const DenseTensor& out_grad,
                            int startup_seed,
                            DenseTensor* x_grad) {
  auto embed_size = out_grad.dims()[out_grad.dims().size() - 1];
  auto elem_size = 1;
  for (auto i = 0; i < out_grad.dims().size() - 1; i++)
    elem_size *= static_cast<int>(out_grad.dims()[i]);

  std::vector<int> idx_vec_grad(elem_size);
  auto* shuffleidx_data = shuffleidx.data<int64_t>();
  for (int i = 0; i < static_cast<int>(idx_vec_grad.size()); i++) {
    idx_vec_grad[shuffleidx_data[i]] = i;
  }

  // copy data according to idx_vec_grad
  auto* out_grad_data = out_grad.data<T>();
  auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);

  for (auto i = 0; i < elem_size; i++) {
    memcpy(x_grad_data + idx_vec_grad[i] * embed_size,
           out_grad_data + i * embed_size,
           embed_size * sizeof(T));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(shuffle_batch_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ShuffleBatchGradKernel,
                   float,
                   double,
                   int32_t,
                   int64_t) {}
