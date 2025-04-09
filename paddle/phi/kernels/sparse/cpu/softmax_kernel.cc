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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/cpu_vec.h"
#include "paddle/phi/kernels/funcs/sparse/softmax.h"
#include "paddle/phi/kernels/softmax_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi::sparse {

template <typename T, typename Context>
void SoftmaxCsrKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      int axis,
                      SparseCsrTensor* out) {
  PADDLE_ENFORCE_EQ(axis,
                    -1,
                    common::errors::Unimplemented(
                        "SparseCsrTensor only support axis=-1 for softmax, "
                        "which is faster when reading data by row (axis=-1)"));
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);
  auto x_dim = x.dims();
  auto x_rank = x_dim.size();

  int batch_size = 1;
  int row_number = 1;
  for (int i = 0; i < x_rank - 1; ++i) {
    if (i < x_rank - 2) {
      batch_size *= static_cast<int>(x_dim[i]);
    } else if (i == x_rank - 2) {
      row_number = static_cast<int>(x_dim[i]);
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
            phi::funcs::vec_add_bias<T, backends::cpu::avx>(
                row_nnz, static_cast<T>(-1) * row_max_val, x_data, out_data);

            phi::funcs::vec_exp<T>(row_nnz, out_data, out_data);

            T sum = 0;
            phi::funcs::vec_sum<T, backends::cpu::avx>(row_nnz, out_data, &sum);
            phi::funcs::vec_scal<T, backends::cpu::avx>(
                row_nnz, static_cast<T>(1) / sum, out_data, out_data);

            x_data = x_data + row_nnz;
            out_data = out_data + row_nnz;
          }
        }
      }));
}

template <typename T, typename IntT, typename Context>
void SoftmaxCooCPUKernel(const Context& dev_ctx,
                         const SparseCooTensor& x,
                         int axis,
                         SparseCooTensor* out) {
  auto indices = x.indices();
  auto values = x.values();
  const auto x_dims = x.dims();
  const auto sparse_dim = x.sparse_dim();
  DenseTensor out_indices(indices);
  DenseTensor out_values = EmptyLike<T, Context>(dev_ctx, values);
  out->SetMember(out_indices, out_values, x.dims(), x.coalesced());

  int dim = axis < 0 ? x_dims.size() + axis : axis;

  /* If dim is greater than or equal to sparse_dim, the dense softmax is used.
   */
  if (dim >= sparse_dim) {
    SoftmaxKernel<T, Context>(
        dev_ctx, values, dim - sparse_dim + 1, &out_values);
    return;
  }

  const std::vector<IntT> sizes = common::vectorize<IntT>(x_dims);
  std::map<IntT, std::vector<IntT>> pools;
  IntT nvalues = std::accumulate(sizes.begin() + sparse_dim,
                                 sizes.end(),
                                 static_cast<IntT>(1),
                                 std::multiplies<>());
  phi::funcs::sparse::GetPoolsSoftmax(out_indices, sizes, dim, &pools);

  auto values_ptr = values.data<T>();
  auto out_values_ptr = out_values.data<T>();
  for (size_t p = 0; p < pools.size(); p++) {
    auto pool_indices = pools[p];
    if (pool_indices.empty()) {
      continue;
    }

    std::vector<T> mx_row(nvalues, -std::numeric_limits<T>::infinity());
    std::vector<T> exp_sums_row(nvalues, 0);
    IntT pool_size = static_cast<IntT>(pool_indices.size());

    // Compute max for each pool
    for (IntT i = 0; i < pool_size; i++) {
      auto values_row = values_ptr + pool_indices[i] * nvalues;
      for (IntT j = 0; j < nvalues; j++) {
        mx_row[j] = std::max(mx_row[j], *(values_row + j));
      }
    }

    // exp to (v - mx) and sum the results
    for (IntT i = 0; i < pool_size; i++) {
      auto values_row = values_ptr + pool_indices[i] * nvalues;
      auto out_values_row = out_values_ptr + pool_indices[i] * nvalues;
      for (IntT j = 0; j < nvalues; j++) {
        auto v = std::exp(*(values_row + j) - mx_row[j]);
        out_values_row[j] = v;
        exp_sums_row[j] += v;
      }
    }

    /* Normalize with the sum of exponents */
    for (IntT i = 0; i < pool_size; i++) {
      auto out_values_row = out_values_ptr + pool_indices[i] * nvalues;
      for (IntT j = 0; j < nvalues; j++) {
        out_values_row[j] *= 1.0 / exp_sums_row[j];
      }
    }
  }
}

// cpu kernel
template <typename T, typename Context>
void SoftmaxCooKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      int axis,
                      SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "SoftmaxCooCPUKernel", ([&] {
        SoftmaxCooCPUKernel<T, data_t, Context>(dev_ctx, x, axis, out);
      }));
}

}  // namespace phi::sparse

PD_REGISTER_KERNEL(softmax_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(softmax_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SoftmaxCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
