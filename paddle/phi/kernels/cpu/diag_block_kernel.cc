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

#include "paddle/phi/kernels/diag_block_kernel.h"
#include <vector>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"

namespace phi {

template <typename T, typename Context>
void DiagBlockKernel(const Context& dev_ctx,
                     const std::vector<const DenseTensor*>& xs,
                     double ss,
                     DenseTensor* out) {
  const phi::DDim& out_dims = out->dims();
  const int num_rows = out_dims[0];
  const int num_cols = out_dims[1];
  EmptyKernel<T, Context>(dev_ctx, {num_rows, num_cols}, out->type(), out);
  T* out_data = out->data<T>();

  int start_i = 0;
  int start_j = 0;
  for (const auto& x : xs) {
    const int x_rows = x->dims()[0];
    const int x_cols = x->dims()[1];
    const T* x_data = x->data<T>();
    // block assign
    for (int i = 0; i < x_rows; i++) {
      for (int j = 0; j < x_cols; j++) {
        out_data[(start_i + i) * num_cols + (start_j + j)] =
            x_data[i * x_cols + j];
      }
    }
    start_i += x_rows;
    start_j += x_cols;
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(diag_block,
                   CPU,
                   ALL_LAYOUT,
                   phi::DiagBlockKernel,
                   bool,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
