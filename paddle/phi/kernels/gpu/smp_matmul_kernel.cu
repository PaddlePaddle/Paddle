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

#include "paddle/phi/kernels/smp_matmul_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SMPMatmulKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     bool x_row_shard,
                     bool y_row_shard,
                     bool out_row_shard,
                     int smp_rank,
                     int smp_ranks,
                     bool trans_y,
                     DenseTensor* out) {
  const std::vector<std::int64_t> x_dims = vectorize(x.dims());
  const std::vector<std::int64_t> y_dims = vectorize(y.dims());

  std::vector<std::int64_t> cal_index;
  for (std::int64_t i = smp_rank; i >= 0; i--) {
    cal_index.push_back(i);
  }
  for (std::int64_t i = smp_ranks - 1; i > smp_rank; i--) {
    cal_index.push_back(i);
  }

  // row_col_col
  if (x_row_shard && !y_row_shard && !out_row_shard) {
    std::int64_t stride = x_dims[x_dims.size() - 1] / smp_ranks;
    std::vector<const DenseTensor*> outputs(smp_ranks);

    for (std::int64_t i = 0; i < cal_index.size(); i++) {
      std::int64_t idx = cal_index[i];
      std::int64_t start = idx * stride;
      std::int64_t end = start + stride;

      DenseTensor dense_out;
      MetaTensor meta_out(&dense_out);
      MatmulInferMeta(x, y, false, trans_y, &meta_out);
      MatmulKernel<T, Context>(dev_ctx, x, y, false, trans_y, &dense_out);

      outputs[idx] = &dense_out;
    }

    ConcatKernel<T, Context>(dev_ctx, outputs, -1, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(smp_matmul,
                   GPU,
                   ALL_LAYOUT,
                   phi::SMPMatmulKernel,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
