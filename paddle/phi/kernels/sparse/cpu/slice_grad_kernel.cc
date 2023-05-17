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

#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void SliceCooGradKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const SparseCooTensor& out_grad,
                        const phi::IntArray& axes_arr,
                        const phi::IntArray& starts_arr,
                        const phi::IntArray& ends_arr,
                        SparseCooTensor* x_grad) {
  const phi::DDim& x_dims = x.dims();

  std::vector<int64_t> axes = axes_arr.GetData();
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();

  // update starts and ends
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(x_dims, &axes, &starts, &ends);

  const int64_t out_grad_nnz = out_grad.nnz();
  auto sparse_dim = static_cast<int64_t>(out_grad.sparse_dim());
  DenseTensor dx_indices =
      phi::Empty<int64_t, Context>(dev_ctx, {sparse_dim, out_grad_nnz});
  DenseTensor dx_values = phi::Empty<T, Context>(dev_ctx, {out_grad_nnz});
  auto* dx_indices_data = dx_indices.data<int64_t>();
  auto* dx_values_data = dx_values.data<T>();

  const auto* out_grad_indices_data = out_grad.indices().data<int64_t>();
  const auto* out_grad_values_data = out_grad.values().data<T>();

  for (int64_t j = 0; j < out_grad_nnz; ++j) {
    // set indices
    for (int64_t i = 0; i < sparse_dim; ++i) {
      dx_indices_data[i * out_grad_nnz + j] =
          out_grad_indices_data[i * out_grad_nnz + j];
    }
    for (size_t ii = 0; ii < axes.size(); ++ii) {
      int64_t i = axes[ii];
      dx_indices_data[i * out_grad_nnz + j] += starts[ii];
    }
    // set value
    dx_values_data[j] = out_grad_values_data[j];
  }

  x_grad->SetMember(dx_indices, dx_values, x.dims(), x.coalesced());
}

template <typename T, typename Context>
void SliceCsrGradKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const SparseCsrTensor& out_grad,
                        const phi::IntArray& axes_arr,
                        const phi::IntArray& starts_arr,
                        const phi::IntArray& ends_arr,
                        SparseCsrTensor* x_grad) {
  const phi::DDim& x_dims = x.dims();

  std::vector<int64_t> axes = axes_arr.GetData();
  std::vector<int64_t> starts = starts_arr.GetData();
  std::vector<int64_t> ends = ends_arr.GetData();

  // update starts and ends
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(x_dims, &axes, &starts, &ends);

  // construct new axes, starts, and ends
  std::vector<int64_t> new_axes(3), new_starts(3), new_ends(3);
  funcs::ConstructNewSliceAttrs(
      x_dims, axes, starts, ends, &new_axes, &new_starts, &new_ends);

  const int64_t out_grad_nnz = out_grad.nnz();
  const int64_t sparse_dim = x_dims.size();

  const auto* out_grad_crows_data = out_grad.crows().data<int64_t>();
  const auto* out_grad_cols_data = out_grad.cols().data<int64_t>();
  const auto* out_grad_values_data = out_grad.values().data<T>();

  if (sparse_dim == 2) {
    const int64_t n_rows = x_dims[0];
    DenseTensor dx_crows = phi::Empty<int64_t>(dev_ctx, {n_rows + 1});
    DenseTensor dx_cols = phi::Empty<int64_t>(dev_ctx, {out_grad_nnz});
    DenseTensor dx_values = phi::Empty<T, Context>(dev_ctx, {out_grad_nnz});
    auto* dx_crows_data = dx_crows.data<int64_t>();
    auto* dx_cols_data = dx_cols.data<int64_t>();
    auto* dx_values_data = dx_values.data<int64_t>();
    // set cols
    for (int64_t i = 0; i < out_grad_nnz; ++i) {
      dx_cols_data[i] = out_grad_cols_data[i] + new_starts[1];
    }
    // set values
    for (int64_t i = 0; i < out_grad_nnz; ++i) {
      dx_values_data[i] = out_grad_values_data[i];
    }
    // set crows
    for (int64_t i = 0; i < new_starts[0]; ++i) {
      dx_crows_data[i] = 0;
    }
    int64_t out_grad_n_rows = out_grad.dims()[0];
    for (int64_t i = 0; i < out_grad_n_rows + 1; ++i) {
      int64_t idx = i + new_starts[0];
      dx_crows_data[idx] = out_grad_crows_data[i];
    }
    for (int64_t i = 0; i < n_rows - new_ends[0]; ++i) {
      int64_t idx = i + new_starts[0] + out_grad_n_rows + 1;
      dx_crows_data[idx] = out_grad_crows_data[out_grad_n_rows - 1];
    }
    x_grad->SetMember(dx_crows, dx_cols, dx_values, x_dims);
  } else if (sparse_dim == 3) {
    const int64_t dim0 = x_dims[0], n_rows = x_dims[1];
    DenseTensor dx_crows = phi::Empty<int64_t>(dev_ctx, {dim0 * (n_rows + 1)});
    DenseTensor dx_cols = phi::Empty<int64_t>(dev_ctx, {out_grad_nnz});
    DenseTensor dx_values = phi::Empty<T, Context>(dev_ctx, {out_grad_nnz});
    auto* dx_crows_data = dx_crows.data<int64_t>();
    auto* dx_cols_data = dx_cols.data<int64_t>();
    auto* dx_values_data = dx_values.data<int64_t>();

    // set cols
    for (int64_t i = 0; i < out_grad_nnz; ++i) {
      dx_cols_data[i] = out_grad_cols_data[i] + new_starts[2];
    }
    // set values
    for (int64_t i = 0; i < out_grad_nnz; ++i) {
      dx_values_data[i] = out_grad_values_data[i];
    }
    // set crows
    int64_t out_grad_n_rows = out_grad.dims()[1];
    for (int64_t i = 0; i < dim0; ++i) {
      if (i < new_starts[0] || i >= new_ends[0]) {
        for (int64_t j = 0; j < n_rows + 1; ++j) {
          dx_crows_data[i * (n_rows + 1) + j] = 0;
        }
      } else {
        int64_t dx_crows_start = i * (n_rows + 1);
        int64_t out_grad_crows_start =
            (i - new_starts[0]) * (out_grad_n_rows + 1);
        for (int64_t j = 0; j < new_starts[1]; ++j) {
          int64_t idx = dx_crows_start + j;
          dx_crows_data[idx] = 0;
        }
        for (int64_t j = 0; j < out_grad_n_rows + 1; ++j) {
          int64_t idx = dx_crows_start + new_starts[1] + j;
          int64_t out_grad_idx = out_grad_crows_start + j;
          dx_crows_data[idx] = out_grad_crows_data[out_grad_idx];
        }
        for (int64_t j = 0; j < n_rows - new_ends[1]; ++j) {
          int64_t idx =
              dx_crows_start + new_starts[1] + out_grad_n_rows + 1 + j;
          int64_t out_grad_idx = out_grad_crows_start + out_grad_n_rows - 1;
          dx_crows_data[idx] = out_grad_crows_data[out_grad_idx];
        }
      }
    }
    x_grad->SetMember(dx_crows, dx_cols, dx_values, x_dims);

  } else {
    // throw exception
    phi::errors::InvalidArgument(
        "Slice grad for Sparse CSR Tensor only support 2-D or 3-D, but got "
        "%d-D.",
        x_dims.size());
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(slice_coo_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SliceCooGradKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(slice_csr_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SliceCsrGradKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
