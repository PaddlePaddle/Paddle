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

#include "paddle/phi/kernels/sparse/softmax_grad_kernel.h"

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
void SoftmaxCsrGradKernel(const Context& dev_ctx,
                          const SparseCsrTensor& out,
                          const SparseCsrTensor& dout,
                          int axis,
                          SparseCsrTensor* dx) {
  PADDLE_ENFORCE_EQ(axis,
                    -1,
                    phi::errors::Unimplemented(
                        "SparseCsrTensor only support axis=-1 for softmax, "
                        "which is faster when reading data by row (axis=-1)"));
  EmptyLikeCsrKernel<T, Context>(dev_ctx, dout, dx);
  auto out_dim = out.dims();
  auto out_rank = out_dim.size();

  int batch_size = 1;
  int row_number = 1;
  for (int i = 0; i < out_rank - 1; ++i) {
    if (i < out_rank - 2) {
      batch_size *= out_dim[i];
    } else if (i == out_rank - 2) {
      row_number = out_dim[i];
    }
  }

  const DenseTensor& out_crows = out.non_zero_crows();
  const DenseTensor& out_values = out.non_zero_elements();
  const DenseTensor& dout_values = dout.non_zero_elements();
  DenseTensor* dx_values = dx->mutable_non_zero_elements();

  int row_nnz = 0;
  const T* out_data = out_values.data<T>();
  const T* dout_data = dout_values.data<T>();
  T* dx_data = dx_values->data<T>();

  // dx = (dout - sum(dout * out)) * out
  PD_VISIT_BASE_INTEGRAL_TYPES(
      out.non_zero_crows().dtype(), "SoftmaxCsrGradKernel", ([&] {
        const data_t* out_crows_data = out_crows.data<data_t>();
        for (int i = 0; i < batch_size; ++i) {
          for (int j = 0; j < row_number; ++j) {
            int crow_idx = i * (row_number + 1) + j;
            row_nnz = static_cast<int>(out_crows_data[crow_idx + 1] -
                                       out_crows_data[crow_idx]);

            T sum = 0;
            phi::funcs::vec_mul_reduce<T, plt::avx>(
                row_nnz, dout_data, out_data, &sum);
            phi::funcs::vec_add_bias<T, plt::avx>(
                row_nnz, static_cast<T>(-1) * sum, dout_data, dx_data);
            phi::funcs::vec_mul<T, plt::avx>(
                row_nnz, dx_data, out_data, dx_data);

            out_data = out_data + row_nnz;
            dout_data = dout_data + row_nnz;
            dx_data = dx_data + row_nnz;
          }
        }
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(softmax_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCsrGradKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
