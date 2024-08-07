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

#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace phi::sparse {

template <typename T, typename Context>
void SliceCooCompute(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     const std::vector<int64_t>& axes,
                     const std::vector<int64_t>& starts,
                     const std::vector<int64_t>& ends,
                     SparseCooTensor* out) {
  const phi::DDim& x_dims = x.dims();

  // Step1: Infer output dims
  auto out_dims = funcs::GetSliceDims<int64_t>(
      x_dims, axes, starts, ends, nullptr, nullptr);

  // Step2: Get out_nnz (the number of non-zero elements in output)
  const int64_t x_nnz = x.nnz();
  int64_t out_nnz = 0;
  const auto* x_indices_data = x.indices().data<int64_t>();
  for (int64_t j = 0; j < x_nnz; ++j) {
    bool hit = true;
    for (size_t ii = 0; ii < axes.size(); ++ii) {
      auto item = x_indices_data[axes[ii] * x_nnz + j];
      if (!(starts[ii] <= item && item < ends[ii])) {
        hit = false;
        break;
      }
    }
    if (!hit) continue;
    out_nnz++;
  }

  // Step3: Get the values and indices of output
  auto sparse_dim = static_cast<int64_t>(x.sparse_dim());
  DenseTensor out_indices =
      phi::Empty<int64_t, Context>(dev_ctx, {sparse_dim, out_nnz});
  DenseTensor out_values = phi::Empty<T, Context>(dev_ctx, {out_nnz});

  auto* out_indices_data = out_indices.data<int64_t>();
  auto* out_values_data = out_values.data<T>();
  const auto* x_values_data = x.values().data<T>();
  int64_t index = 0;
  for (int64_t j = 0; j < x_nnz && index < out_nnz; ++j) {
    bool hit = true;
    for (size_t ii = 0; ii < axes.size(); ++ii) {
      auto item = x_indices_data[axes[ii] * x_nnz + j];
      if (!(starts[ii] <= item && item < ends[ii])) {
        hit = false;
        break;
      }
    }
    if (!hit) continue;
    // set value
    out_values_data[index] = x_values_data[j];
    // set coordinate
    for (int64_t i = 0; i < sparse_dim; ++i) {
      out_indices_data[i * out_nnz + index] = x_indices_data[i * x_nnz + j];
    }
    for (size_t ii = 0; ii < axes.size(); ++ii) {
      auto i = axes[ii];
      out_indices_data[i * out_nnz + index] -= starts[ii];
    }
    index++;
  }
  out->SetMember(out_indices, out_values, out_dims, x.coalesced());
}

template <typename T, typename Context>
void SliceCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const phi::IntArray& axes,
                    const phi::IntArray& starts,
                    const phi::IntArray& ends,
                    SparseCooTensor* out) {
  const phi::DDim& x_dims = x.dims();
  std::vector<int64_t> axes_vec = axes.GetData();
  std::vector<int64_t> starts_vec = starts.GetData();
  std::vector<int64_t> ends_vec = ends.GetData();

  // Check and update attr
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(
      x_dims, &axes_vec, &starts_vec, &ends_vec);

  SliceCooCompute<T, Context>(dev_ctx, x, axes_vec, starts_vec, ends_vec, out);
}

int64_t GetCsrNonZeroNumber(const SparseCsrTensor& x,
                            const int64_t x_crows_start,
                            const int64_t x_crows_end,
                            const int64_t min_col,
                            const int64_t max_col,
                            const int64_t x_cols_offset = 0) {
  const auto* x_crows_data = x.crows().data<int64_t>();
  const auto* x_cols_data = x.cols().data<int64_t>();
  int64_t out_nnz = 0;
  for (int64_t i = x_crows_start; i < x_crows_end; ++i) {
    int64_t st = x_crows_data[i] + x_cols_offset;
    int64_t ed = x_crows_data[i + 1] + x_cols_offset;
    for (int64_t jj = st; jj < ed; ++jj) {
      if (x_cols_data[jj] >= min_col && x_cols_data[jj] < max_col) {
        out_nnz++;
      }
    }
  }
  return out_nnz;
}

template <typename T>
void GetCsrSubMatrix(const SparseCsrTensor& x,
                     const int64_t x_crows_start,
                     const int64_t x_crows_end,
                     const int64_t min_col,
                     const int64_t max_col,
                     DenseTensor* out_crows,
                     DenseTensor* out_cols,
                     DenseTensor* out_values,
                     const int64_t x_cols_offset = 0,
                     const int64_t out_crows_offset = 0,
                     const int64_t out_cols_offset = 0) {
  const auto* x_crows_data = x.crows().data<int64_t>();
  const auto* x_cols_data = x.cols().data<int64_t>();
  const auto* x_values_data = x.values().data<T>();

  auto* out_crows_data = out_crows->data<int64_t>();
  auto* out_cols_data = out_cols->data<int64_t>();
  auto* out_values_data = out_values->data<T>();
  out_crows_data[out_crows_offset] = 0;
  int64_t index = 0, out_n_rows = x_crows_end - x_crows_start;
  for (int i = 0; i < out_n_rows; ++i) {
    int64_t st = x_crows_data[x_crows_start + i] + x_cols_offset;
    int64_t ed = x_crows_data[x_crows_start + i + 1] + x_cols_offset;
    for (int64_t jj = st; jj < ed; ++jj) {
      if (x_cols_data[jj] >= min_col && x_cols_data[jj] < max_col) {
        out_cols_data[out_cols_offset + index] = x_cols_data[jj] - min_col;
        out_values_data[out_cols_offset + index] = x_values_data[jj];
        index++;
      }
    }
    out_crows_data[out_crows_offset + i + 1] = index;
  }
}

template <typename T, typename Context>
void SliceCsrTensor2D(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const std::vector<int64_t>& axes,
                      const std::vector<int64_t>& starts,
                      const std::vector<int64_t>& ends,
                      const phi::DDim& out_dims,
                      SparseCsrTensor* out) {
  // Step1: Get nnz of out
  int64_t out_nnz =
      GetCsrNonZeroNumber(x, starts[0], ends[0], starts[1], ends[1], 0);
  // Step2: Set out
  int64_t out_n_rows = ends[0] - starts[0];
  DenseTensor out_crows =
      phi::Empty<int64_t, Context>(dev_ctx, {out_n_rows + 1});
  DenseTensor out_cols = phi::Empty<int64_t, Context>(dev_ctx, {out_nnz});
  DenseTensor out_values = phi::Empty<T, Context>(dev_ctx, {out_nnz});
  GetCsrSubMatrix<T>(x,
                     starts[0],
                     ends[0],
                     starts[1],
                     ends[1],
                     &out_crows,
                     &out_cols,
                     &out_values,
                     0,
                     0,
                     0);
  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename Context>
void SliceCsrTensor3D(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const std::vector<int64_t>& axes,
                      const std::vector<int64_t>& starts,
                      const std::vector<int64_t>& ends,
                      const phi::DDim& out_dims,
                      SparseCsrTensor* out) {
  const auto* x_crows_data = x.crows().data<int64_t>();
  // Step1: Get nnz of out
  const int64_t x_dim0 = x.dims()[0], x_n_rows = x.dims()[1];
  int64_t x_cols_offset = 0, out_nnz = 0;
  // all_nnzs stores the nnz along with out_dim0, which will be used to set out.
  std::vector<int64_t> all_nnzs(ends[0] - starts[0]);
  for (int64_t i = 0; i < x_dim0; ++i) {
    if (i >= starts[0] && i < ends[0]) {  // slice dim 0
      int64_t x_crows_st = i * (x_n_rows + 1) + starts[1];
      int64_t x_crows_ed = i * (x_n_rows + 1) + ends[1];
      int64_t nnz = GetCsrNonZeroNumber(
          x, x_crows_st, x_crows_ed, starts[2], ends[2], x_cols_offset);
      out_nnz += nnz;
      all_nnzs[i - starts[0]] = nnz;
    }
    // get the start index in non_zero_cols_
    x_cols_offset += x_crows_data[(i + 1) * (x_n_rows + 1) - 1];
  }

  // Step2: Set out
  const int64_t out_dim0 = out_dims[0], out_n_rows = out_dims[1];
  DenseTensor out_crows =
      phi::Empty<int64_t, Context>(dev_ctx, {out_dim0 * (out_n_rows + 1)});
  DenseTensor out_cols = phi::Empty<int64_t, Context>(dev_ctx, {out_nnz});
  DenseTensor out_values = phi::Empty<T, Context>(dev_ctx, {out_nnz});

  x_cols_offset = 0;
  int64_t out_crows_offset = 0, out_cols_offset = 0;
  for (int64_t i = 0; i < x_dim0; ++i) {
    if (i >= starts[0] && i < ends[0]) {  // slice dim 0
      int64_t x_crows_start = i * (x_n_rows + 1) + starts[1];
      int64_t x_crows_end = i * (x_n_rows + 1) + ends[1];
      GetCsrSubMatrix<T>(x,
                         x_crows_start,
                         x_crows_end,
                         starts[2],
                         ends[2],
                         &out_crows,
                         &out_cols,
                         &out_values,
                         x_cols_offset,
                         out_crows_offset,
                         out_cols_offset);
      out_crows_offset += (out_n_rows + 1);
      out_cols_offset += all_nnzs[i - starts[0]];
    }
    x_cols_offset += x_crows_data[(i + 1) * (x_n_rows + 1) - 1];
  }
  out->SetMember(out_crows, out_cols, out_values, out_dims);
}

template <typename T, typename Context>
void SliceCsrCompute(const Context& dev_ctx,
                     const SparseCsrTensor& x,
                     const std::vector<int64_t>& axes,
                     const std::vector<int64_t>& starts,
                     const std::vector<int64_t>& ends,
                     SparseCsrTensor* out) {
  const phi::DDim& x_dims = x.dims();

  // Step1: Infer output dims
  auto out_dims = funcs::GetSliceDims<int64_t>(
      x_dims, axes, starts, ends, nullptr, nullptr);

  // Step2: Construct new axes, starts and ends.
  std::vector<int64_t> new_axes(3), new_starts(3), new_ends(3);
  funcs::ConstructNewSliceAttrs(
      x_dims, axes, starts, ends, &new_axes, &new_starts, &new_ends);

  // Setp3: Slice csr tensor according to its dimension
  if (x_dims.size() == 2) {
    SliceCsrTensor2D<T, Context>(
        dev_ctx, x, new_axes, new_starts, new_ends, out_dims, out);
  } else if (x_dims.size() == 3) {
    SliceCsrTensor3D<T, Context>(
        dev_ctx, x, new_axes, new_starts, new_ends, out_dims, out);
  } else {
    // throw exception
    common::errors::InvalidArgument(
        "Slice for Sparse CSR Tensor only support 2-D or 3-D, but got %d-D.",
        x_dims.size());
  }
}

template <typename T, typename Context>
void SliceCsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    const phi::IntArray& axes,
                    const phi::IntArray& starts,
                    const phi::IntArray& ends,
                    SparseCsrTensor* out) {
  const phi::DDim& x_dims = x.dims();
  std::vector<int64_t> axes_vec = axes.GetData();
  std::vector<int64_t> starts_vec = starts.GetData();
  std::vector<int64_t> ends_vec = ends.GetData();

  // Check and update attr
  funcs::CheckAndUpdateSparseSliceAttrs<int64_t>(
      x_dims, &axes_vec, &starts_vec, &ends_vec);
  SliceCsrCompute<T, Context>(dev_ctx, x, axes_vec, starts_vec, ends_vec, out);
}
}  // namespace phi::sparse

PD_REGISTER_KERNEL(slice_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SliceCooKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(slice_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SliceCsrKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
