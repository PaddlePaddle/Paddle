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

#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi::sparse {

template <typename T, typename Context>
void TransposeCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const std::vector<int>& perm,
                        SparseCooTensor* out) {
  // create out sparse tensor
  int64_t x_nnz = x.nnz();
  DDim out_dims = x.dims().transpose(perm);
  DenseTensor out_indices = EmptyLike<int64_t, Context>(dev_ctx, x.indices());
  const DenseTensor& out_values(x.values());
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());

  // compute values of indices
  const DenseTensor& x_indices = x.indices();
  const auto* x_indices_data = x_indices.data<int64_t>();
  auto* out_indices_data = out_indices.data<int64_t>();
  for (unsigned int i = 0; i < perm.size(); ++i) {
    for (int64_t j = 0; j < x_nnz; ++j) {
      out_indices_data[j + i * x_nnz] = x_indices_data[j + perm[i] * x_nnz];
    }
  }
}

template <typename T, typename Context>
void TransposeCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const std::vector<int>& perm,
                        SparseCsrTensor* out) {
  unsigned int n_dim = perm.size();
  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_cols = x.cols();
  const DenseTensor& x_values = x.values();
  DenseTensor out_crows, out_cols, out_values;
  // return a copy of x
  if (perm[0] == 0 && perm[1] == 1 && (n_dim == 2 || perm[2] == 2)) {
    out_crows = x_crows;
    out_cols = x_cols;
    out_values = x_values;
    out->SetMember(out_crows, out_cols, out_values, x.dims());
    return;
  }
  // create out sparse tensor
  DDim out_dims = x.dims().transpose(perm);
  if (n_dim == 2) {
    out_crows = Empty<int64_t, Context>(dev_ctx, {out_dims[0] + 1});
  } else {
    out_crows =
        Empty<int64_t, Context>(dev_ctx, {out_dims[0] * (out_dims[1] + 1)});
  }
  out_cols = EmptyLike<int64_t, Context>(dev_ctx, x.cols());
  out_values = EmptyLike<T, Context>(dev_ctx, x.values());
  out->SetMember(out_crows, out_cols, out_values, out_dims);
  // transpose by two stages
  if (perm[0] == 1 && perm[1] == 2) {  // perm == {1, 2, 0}
    SparseCsrTensor temp;
    TransposeCsrKernel<T, Context>(dev_ctx, x, {1, 0, 2}, &temp);
    TransposeCsrKernel<T, Context>(dev_ctx, temp, {0, 2, 1}, out);
    return;
  } else if (perm[0] == 2 && perm[1] == 0) {  // perm == {2, 0, 1}
    SparseCsrTensor temp;
    TransposeCsrKernel<T, Context>(dev_ctx, x, {0, 2, 1}, &temp);
    TransposeCsrKernel<T, Context>(dev_ctx, temp, {1, 0, 2}, out);
    return;
  } else if (perm[0] == 2 && perm[1] == 1) {  // perm == {2, 1, 0}
    SparseCsrTensor temp;
    TransposeCsrKernel<T, Context>(dev_ctx, x, {1, 0, 2}, &temp);
    TransposeCsrKernel<T, Context>(dev_ctx, temp, {2, 0, 1}, out);
    return;
  }

  int64_t* out_crows_data = out_crows.data<int64_t>();
  int64_t* out_cols_data = out_cols.data<int64_t>();
  T* out_values_data = out_values.data<T>();
  const int64_t* x_crows_data = x_crows.data<int64_t>();
  const int64_t* x_cols_data = x_cols.data<int64_t>();
  const T* x_values_data = x_values.data<T>();

  int64_t x_nnz = x.nnz();
  if (n_dim == 2) {  // perm == {1, 0}
    // compute out_crows_data by x_cols_data
    for (int i = 0; i < out_dims[0]; ++i) {
      out_crows_data[i] = 0;
    }
    for (int i = 0; i < x_nnz; ++i) {
      int64_t j = x_cols_data[i];
      out_crows_data[j + 1]++;
    }
    out_crows_data[out_dims[0]] = x_nnz;
    for (int i = 1; i < out_dims[0]; ++i) {
      out_crows_data[i] += out_crows_data[i - 1];
    }
    // compute out_cols_data and out_values_data by out_crows_data and x
    std::unordered_map<int64_t, int> cols_offset;
    for (int i = 0; i < x.dims()[0]; ++i) {
      int64_t start = x_crows_data[i];
      int64_t end = x_crows_data[i + 1];
      for (int64_t j = start; j < end; ++j) {
        int64_t x_cols_j = x_cols_data[j];
        int64_t jjj = out_crows_data[x_cols_j];
        if (cols_offset.count(jjj)) {
          cols_offset[jjj]++;
        } else {
          cols_offset[jjj] = 0;
        }
        int64_t jjj_offset = jjj + cols_offset[jjj];
        out_cols_data[jjj_offset] = i;
        out_values_data[jjj_offset] = x_values_data[j];
      }
    }
  } else {  // n_dim == 3
    int64_t out_n_rows = out_dims[1];
    int64_t x_n_rows = x.dims()[1];
    for (int k = 0; k < out_dims[0]; ++k) {
      if (perm[0] == 0) {  // perm == {0, 2, 1}
        // compute out_crows_data by x_cols_data
        for (int i = 0; i < out_n_rows; ++i) {
          out_crows_data[i] = 0;
        }
        for (int i = 0; i < x_crows_data[x_n_rows]; ++i) {
          int64_t j = x_cols_data[i];
          out_crows_data[j + 1]++;
        }
        out_crows_data[out_n_rows] = x_crows_data[x_n_rows];
        for (int i = 1; i < out_n_rows; ++i) {
          out_crows_data[i] += out_crows_data[i - 1];
        }
        // compute out_cols_data and out_values_data by out_crows_data and x
        std::unordered_map<int64_t, int> cols_offset;
        for (int i = 0; i < x_n_rows; ++i) {
          int64_t start = x_crows_data[i];
          int64_t end = x_crows_data[i + 1];
          for (int64_t j = start; j < end; ++j) {
            int64_t x_cols_j = x_cols_data[j];
            int64_t jjj = out_crows_data[x_cols_j];
            if (cols_offset.count(jjj)) {
              cols_offset[jjj]++;
            } else {
              cols_offset[jjj] = 0;
            }
            int64_t jjj_offset = jjj + cols_offset[jjj];
            out_cols_data[jjj_offset] = i;
            out_values_data[jjj_offset] = x_values_data[j];
          }
        }
        // x offset
        x_cols_data += x_crows_data[x_n_rows];
        x_values_data += x_crows_data[x_n_rows];
        x_crows_data += x_n_rows + 1;
      } else if (perm[0] == 1 && perm[1] == 0) {  // perm == {1, 0, 2}
        for (int i = 0; i < out_n_rows; ++i) {
          out_crows_data[i] = 0;
        }
        int64_t x_cols_offset = 0;
        int out_cols_index = 0;
        for (int i = 0; i < x.dims()[0]; ++i) {
          int x_crows_index = static_cast<int>(i * (x_n_rows + 1));
          int64_t start = x_crows_data[x_crows_index + k];
          int64_t end = x_crows_data[x_crows_index + 1 + k];
          out_crows_data[i + 1] = end - start;
          for (int64_t j = start; j < end; ++j) {
            out_cols_data[out_cols_index] = x_cols_data[x_cols_offset + j];
            out_values_data[out_cols_index] = x_values_data[x_cols_offset + j];
            out_cols_index++;
          }
          x_cols_offset += x_crows_data[x_crows_index + x_n_rows];
        }
        for (int i = 1; i <= out_n_rows; ++i) {
          out_crows_data[i] += out_crows_data[i - 1];
        }
      }
      // out offset
      out_cols_data += out_crows_data[out_n_rows];
      out_values_data += out_crows_data[out_n_rows];
      out_crows_data += out_n_rows + 1;
    }
  }
}
}  // namespace phi::sparse

PD_REGISTER_KERNEL(transpose_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::TransposeCooKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(transpose_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::TransposeCsrKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
