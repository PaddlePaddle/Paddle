/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/softmax_kernel.h"

#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/cpu_vec.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace plt = paddle::platform;

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SoftmaxCsrKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      int axis,
                      SparseCsrTensor* out) {
  PADDLE_ENFORCE_EQ(axis,
                    -1,
                    phi::errors::Unimplemented(
                        "SparseCsrTensor only support axis=-1 for softmax, "
                        "which is faster when reading data by row (axis=-1)"));
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);
  auto x_dim = x.dims();
  auto x_rank = x_dim.size();

  int batch_size = 1;
  int row_number = 1;
  for (int i = 0; i < x_rank - 1; ++i) {
    if (i < x_rank - 2) {
      batch_size *= x_dim[i];
    } else if (i == x_rank - 2) {
      row_number = x_dim[i];
    }
  }

  const DenseTensor& x_crows = x.non_zero_crows();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_values = out->mutable_non_zero_elements();

  int row_nnz = 0;
  T row_max_val = 0;
  const T* x_data = x_values.data<T>();
  T* out_data = out_values->data<T>();

  // out = exp(x-x_max) / sum( exp(x-x_max ))
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.non_zero_crows().dtype(), "CsrSoftmaxKernel", ([&] {
        const data_t* x_crows_data = x_crows.data<data_t>();
        for (int i = 0; i < batch_size; ++i) {
          for (int j = 0; j < row_number; ++j) {
            int crow_idx = i * (row_number + 1) + j;
            row_nnz = static_cast<int>(x_crows_data[crow_idx + 1] -
                                       x_crows_data[crow_idx]);

            row_max_val = *std::max_element(x_data, x_data + row_nnz);
            phi::funcs::vec_add_bias<T, plt::avx>(
                row_nnz, static_cast<T>(-1) * row_max_val, x_data, out_data);

            phi::funcs::vec_exp<T>(row_nnz, out_data, out_data);

            T sum = 0;
            phi::funcs::vec_sum<T, plt::avx>(row_nnz, out_data, &sum);
            phi::funcs::vec_scal<T, plt::avx>(
                row_nnz, static_cast<T>(1) / sum, out_data, out_data);

            x_data = x_data + row_nnz;
            out_data = out_data + row_nnz;
          }
        }
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(softmax_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
