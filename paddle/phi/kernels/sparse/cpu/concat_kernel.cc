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

#include "paddle/phi/kernels/sparse/concat_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace sparse {

static void check_cat_sparse_dims(const SparseCooTensor* t,
                                  int64_t pos,
                                  DDim dims,
                                  int64_t axis,
                                  int64_t sparse_dim,
                                  int64_t dense_dim) {
  PADDLE_ENFORCE_EQ(t->sparse_dim(),
                    sparse_dim,
                    "All tensors must have the same sparse_dim ",
                    sparse_dim,
                    ", but tensor at position ",
                    pos,
                    " has ",
                    t->sparse_dim());
  PADDLE_ENFORCE_EQ(t->dense_dim(),
                    dense_dim,
                    "All tensors must have the same dense_dim ",
                    dense_dim,
                    ", but tensor at position ",
                    pos,
                    " has ",
                    t->dense_dim());
}

template <typename T, typename Context>
void ConcatCsr3D2A(const std::vector<const phi::SparseCsrTensor*>& x,
                   const Context& dev_ctx,
                   const size_t num_split,
                   const std::vector<const int64_t*>& crows_data_vec,
                   int64_t* out_cols_data,
                   const std::vector<const int64_t*>& cols_data_vec,
                   T* out_values_data,
                   const std::vector<const T*>& values_data_vec,
                   phi::SparseCsrTensor* out,
                   phi::DenseTensor* out_cols,
                   phi::DenseTensor* out_values,
                   phi::DDim* out_dims) {
  int64_t batch = static_cast<int>(x[0]->dims()[0]);
  int64_t rows = static_cast<int>(x[0]->dims()[1]);
  int64_t now_crow_numel = rows + 1;

  DenseTensor out_crows =
      phi::Empty<int64_t>(dev_ctx, {now_crow_numel * batch});
  int64_t* out_crows_data = out_crows.data<int64_t>();

  const int64_t* now_crow_ptr = nullptr;
  int64_t cumulative_offset = 0;
  int64_t column_offset = 0;
  std::vector<int64_t> offset_vec(num_split, 0);
  for (int64_t b = 0; b < batch; b++) {
    out_crows_data[0] = 0;
    cumulative_offset = 0;
    for (int64_t j = 1; j < now_crow_numel; j++) {
      column_offset = 0;
      for (size_t i = 0; i < num_split; i++) {
        now_crow_ptr = crows_data_vec[i] + b * now_crow_numel;

        for (int64_t k = 0; k < now_crow_ptr[j] - now_crow_ptr[j - 1]; k++) {
          *out_cols_data = cols_data_vec[i][offset_vec[i]] + column_offset;
          *out_values_data = values_data_vec[i][offset_vec[i]];
          offset_vec[i]++;
          out_cols_data++;
          out_values_data++;
          cumulative_offset++;
        }
        column_offset += static_cast<size_t>(x[i]->dims()[2]);
        out_crows_data[j] = cumulative_offset;
      }
    }
    out_crows_data += now_crow_numel;
  }

  out->SetMember(out_crows, *out_cols, *out_values, *out_dims);
}

template <typename T, typename Context>
void ConcatCsr3D1A(const std::vector<const phi::SparseCsrTensor*>& x,
                   int64_t out_crows_size,
                   const size_t num_split,
                   const Context& dev_ctx,
                   const std::vector<const int64_t*>& crows_data_vec,
                   const std::vector<const T*>& values_data_vec,
                   const std::vector<const int64_t*>& cols_data_vec,
                   T* out_values_data,
                   int64_t* out_cols_data,
                   phi::SparseCsrTensor* out,
                   phi::DenseTensor* out_cols,
                   phi::DenseTensor* out_values,
                   phi::DDim* out_dims) {
  std::vector<int64_t> crows_numel;

  size_t batch = static_cast<int>(x[0]->dims()[0]);
  // out_crows need 0 for each round of batch
  out_crows_size = batch;
  for (size_t i = 0; i < num_split; i++) {
    int64_t rows = static_cast<int64_t>(x[i]->dims()[1]);
    crows_numel.push_back(rows + 1);
    out_crows_size += batch * rows;
  }
  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {out_crows_size});
  int64_t* out_crows_data = out_crows.data<int64_t>();
  int64_t value_offset = 0, crow_index = 0, cumulative_offset = 0;
  const T* now_value_ptr = nullptr;
  const int64_t* now_cols_ptr = nullptr;
  const int64_t* now_crows_ptr = nullptr;
  std::vector<int64_t> values_index(num_split + 1, 0);
  auto cpu_place = dev_ctx.GetPlace();
  for (size_t b = 0; b < batch; b++) {
    out_crows_data[crow_index] = 0;
    crow_index++;
    cumulative_offset = 0;

    for (size_t i = 0; i < num_split; i++) {
      const int64_t* x_crows_ptr = x[i]->crows().data<int64_t>();
      // nnz for batch and in tensor
      int64_t x_crows_nnz = x_crows_ptr[(b + 1) * (crows_numel[i]) - 1];
      now_crows_ptr = crows_data_vec[i] + b * crows_numel[i];
      now_value_ptr = values_data_vec[i] + values_index[i];
      now_cols_ptr = cols_data_vec[i] + values_index[i];
      values_index[i] += x_crows_nnz;

      if (x_crows_nnz) {
        memory_utils::Copy(cpu_place,
                           out_values_data + value_offset,
                           cpu_place,
                           now_value_ptr,
                           x_crows_nnz * sizeof(T));
        memory_utils::Copy(cpu_place,
                           out_cols_data + value_offset,
                           cpu_place,
                           now_cols_ptr,
                           x_crows_nnz * sizeof(int64_t));
      }
      value_offset += x_crows_nnz;
      for (int64_t j = 1; j < crows_numel[i]; j++) {
        out_crows_data[crow_index] = now_crows_ptr[j] + cumulative_offset;
        crow_index++;
      }

      cumulative_offset += now_crows_ptr[crows_numel[i] - 1];
    }
  }
  out->SetMember(out_crows, *out_cols, *out_values, *out_dims);
}

template <typename T, typename Context>
void ConcatCsr3D0A(const size_t num_split,
                   const std::vector<const phi::SparseCsrTensor*>& x,
                   int64_t out_crows_size,
                   const Context& dev_ctx,
                   phi::SparseCsrTensor* out,
                   phi::DenseTensor* out_values,
                   phi::DenseTensor* out_cols,
                   phi::DDim* out_dims) {
  std::vector<DenseTensor> crows;
  std::vector<DenseTensor> values;
  std::vector<DenseTensor> cols;

  for (size_t i = 0; i < num_split; i++) {
    crows.emplace_back(x[i]->crows());
    values.emplace_back(x[i]->values());
    cols.emplace_back(x[i]->cols());
    out_crows_size += x[i]->crows().numel();
  }

  phi::sparse::ConcatFunctor<T, Context>(
      dev_ctx, values, static_cast<T>(0), out_values);

  phi::sparse::ConcatFunctor<int64_t, Context>(
      dev_ctx, cols, static_cast<int64_t>(0), out_cols);
  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {out_crows_size});
  phi::sparse::ConcatFunctor<int64_t, Context>(
      dev_ctx, crows, static_cast<int64_t>(0), &out_crows);

  out->SetMember(out_crows, *out_cols, *out_values, *out_dims);
}

template <typename T, typename Context>
void ConcatCsr2D1A(const std::vector<const phi::SparseCsrTensor*>& x,
                   int64_t out_crows_size,
                   const Context& dev_ctx,
                   const size_t num_split,
                   const std::vector<const int64_t*>& crows_data_vec,
                   int64_t* out_cols_data,
                   const std::vector<const int64_t*>& cols_data_vec,
                   T* out_values_data,
                   const std::vector<const T*>& values_data_vec,
                   phi::SparseCsrTensor* out,
                   phi::DenseTensor* out_cols,
                   phi::DenseTensor* out_values,
                   phi::DDim* out_dims) {
  int64_t rows = static_cast<size_t>(x[0]->dims()[0]);
  out_crows_size = rows + 1;

  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {out_crows_size});
  int64_t* out_crows_data = out_crows.data<int64_t>();
  out_crows_data[0] = 0;
  int64_t out_index = 0;

  std::vector<int64_t> offset_vec(num_split, 0);
  int64_t column_offset = 0;
  for (int64_t j = 1; j < out_crows_size; j++) {
    column_offset = 0;
    for (size_t i = 0; i < num_split; i++) {
      for (int64_t k = 0; k < crows_data_vec[i][j] - crows_data_vec[i][j - 1];
           k++) {
        out_cols_data[out_index] =
            cols_data_vec[i][offset_vec[i]] + column_offset;
        out_values_data[out_index] = values_data_vec[i][offset_vec[i]];
        offset_vec[i]++;
        out_index++;
      }
      out_crows_data[j] = out_index;
      column_offset += static_cast<size_t>(x[i]->dims()[1]);
    }
  }

  out->SetMember(out_crows, *out_cols, *out_values, *out_dims);
}

template <typename T, typename Context>
void ConcatCsr2D0A(int64_t out_crows_size,
                   const size_t num_split,
                   const std::vector<const phi::SparseCsrTensor*>& x,
                   const Context& dev_ctx,
                   T* out_values_data,
                   const std::vector<const T*>& values_data_vec,
                   int64_t* out_cols_data,
                   const std::vector<const int64_t*>& cols_data_vec,
                   const std::vector<const int64_t*>& crows_data_vec,
                   phi::SparseCsrTensor* out,
                   phi::DenseTensor* out_cols,
                   phi::DenseTensor* out_values,
                   phi::DDim* out_dims) {
  std::vector<int64_t> nnz_vec;
  std::vector<int64_t> crows_numel;
  out_crows_size = 1;
  for (size_t i = 0; i < num_split; i++) {
    int64_t rows = static_cast<int>(x[i]->dims()[0]);
    crows_numel.push_back(rows + 1);
    out_crows_size += rows;
    nnz_vec.push_back(x[i]->nnz());
  }

  DenseTensor out_crows = phi::Empty<int64_t>(dev_ctx, {out_crows_size});
  int64_t* out_crows_data = out_crows.data<int64_t>();

  int64_t value_offset = 0;
  auto cpu_place = dev_ctx.GetPlace();
  for (size_t i = 0; i < num_split; i++) {
    int nnz = nnz_vec[i];
    if (nnz) {
      memory_utils::Copy(cpu_place,
                         out_values_data + value_offset,
                         cpu_place,
                         values_data_vec[i],
                         nnz * sizeof(T));
      memory_utils::Copy(cpu_place,
                         out_cols_data + value_offset,
                         cpu_place,
                         cols_data_vec[i],
                         nnz * sizeof(int64_t));
      value_offset += nnz;
    }
  }

  out_crows_data[0] = 0;

  int64_t cumulative_offset = 0;
  int64_t out_index = 1;
  for (size_t i = 0; i < num_split; i++) {
    for (int64_t j = 1; j < crows_numel[i]; j++) {
      out_crows_data[out_index] = crows_data_vec[i][j] + cumulative_offset;
      out_index++;
    }
    cumulative_offset += crows_data_vec[i][crows_numel[i] - 1];
  }
  out->SetMember(out_crows, *out_cols, *out_values, *out_dims);
}

template <typename T, typename Context>
void ConcatCooKernel(const Context& dev_ctx,
                     const std::vector<const SparseCooTensor*>& x,
                     const Scalar& axis_scalar,
                     SparseCooTensor* out) {
  std::vector<DenseTensor> indices;
  std::vector<DenseTensor> values;
  std::vector<phi::DDim> x_dims;

  int64_t axis = axis_scalar.to<int64_t>();
  axis = phi::funcs::ComputeAxis(axis, x[0]->dims().size());
  int32_t sparse_dim = x[0]->sparse_dim();
  int32_t dense_dim = x[0]->dense_dim();

  DDim dims = x[0]->dims();
  DenseTensor out_indices;
  DenseTensor out_values;

  int64_t pos = 0;
  for (const auto* t : x) {
    check_cat_sparse_dims(t, pos, dims, axis, sparse_dim, dense_dim);
    x_dims.push_back(t->dims());
    pos++;
  }

  EmptyLikeCooKernel<T, Context>(dev_ctx, *x[0], out);
  phi::DDim out_dims = phi::funcs::ComputeAndCheckShape(true, x_dims, axis);
  if (axis < sparse_dim) {
    int64_t out_nnz = 0, out_cols = 0;
    std::vector<int64_t> indice_offset;
    indice_offset.push_back(out_cols);
    for (const auto* t : x) {
      indices.emplace_back(t->indices());
      values.emplace_back(t->values());
      out_nnz += t->nnz();
      out_cols += t->dims()[axis];
      indice_offset.push_back(out_cols);
    }
    out_indices = phi::Empty<int64_t, Context>(dev_ctx, {sparse_dim, out_nnz});

    DDim v_dim = x[0]->values().dims();
    v_dim[0] = out_nnz;
    IntArray v_shape(v_dim.GetMutable(), v_dim.size());
    out_values = phi::Empty<T, Context>(dev_ctx, v_shape);
    // TODO(bapijun) use a std::vector<DenseTensor *> to optimize the
    // concat_functor
    phi::sparse::ConcatFunctor<int64_t, Context>(
        dev_ctx, indices, static_cast<int>(1), &out_indices);

    phi::sparse::ConcatFunctor<T, Context>(
        dev_ctx, values, static_cast<int>(0), &out_values);

    int64_t col = 0;
    auto* out_indices_data = out_indices.data<int64_t>();
    for (size_t i = 0; i < x.size(); i++) {
      int64_t piece_size = x[i]->nnz();
      if (i > 0) {
        for (int64_t j = col; j < col + piece_size; j++) {
          out_indices_data[axis * out_nnz + j] += indice_offset[i];
        }
      }
      col += piece_size;
    }

    out->SetMember(out_indices, out_values, out_dims, x[0]->coalesced());

    return;
  } else {
    int64_t values_dim = axis - sparse_dim + 1;
    int64_t total_size = 0;
    for (auto& r : x) {
      total_size += r->values().dims()[values_dim];
    }
    DDim zeros_sizes = x[0]->values().dims();
    int64_t cumulative_size = 0;

    for (const auto* t : x) {
      zeros_sizes[0] = t->values().dims()[0];
      zeros_sizes[values_dim] = cumulative_size;
      cumulative_size += t->values().dims()[values_dim];
      // z1,z2 is a vector of all zeros
      DenseTensor z1 =
          phi::Full<T, Context>(dev_ctx, common::vectorize(zeros_sizes), 0);
      zeros_sizes[values_dim] = total_size - cumulative_size;
      DenseTensor z2 =
          phi::Full<T, Context>(dev_ctx, common::vectorize(zeros_sizes), 0);
      std::vector<DenseTensor> now_values;
      now_values.emplace_back(z1);
      now_values.emplace_back(t->values());
      now_values.emplace_back(z2);
      DenseTensor concat_value =
          phi::Empty<T, Context>(dev_ctx, common::vectorize(zeros_sizes));
      phi::sparse::ConcatFunctor<T, Context>(
          dev_ctx, now_values, values_dim, &concat_value);
      values.emplace_back(concat_value);
      indices.emplace_back(t->indices());
    }
    phi::sparse::ConcatFunctor<int64_t, Context>(
        dev_ctx, indices, static_cast<int>(1), &out_indices);

    phi::sparse::ConcatFunctor<T, Context>(
        dev_ctx, values, static_cast<int>(0), &out_values);

    out->SetMember(out_indices, out_values, out_dims, x[0]->coalesced());
    return;
  }
}

template <typename T, typename Context>
void ConcatCsrKernel(const Context& dev_ctx,
                     const std::vector<const SparseCsrTensor*>& x,
                     const Scalar& axis_scalar,
                     SparseCsrTensor* out) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, *x[0], out);
  const size_t num_split = x.size();
  std::vector<phi::DDim> x_dims;
  x_dims.reserve(num_split);

  int64_t axis = axis_scalar.to<int64_t>();
  axis = phi::funcs::ComputeAxis(axis, x[0]->dims().size());

  std::vector<const int64_t*> crows_data_vec;
  std::vector<const T*> values_data_vec;
  std::vector<const int64_t*> cols_data_vec;

  int64_t out_values_size = 0;
  int64_t out_crows_size = 0;
  for (const auto* t : x) {
    x_dims.emplace_back(t->dims());
    values_data_vec.push_back(t->values().data<T>());
    cols_data_vec.push_back(t->cols().data<int64_t>());
    crows_data_vec.push_back(t->crows().data<int64_t>());
    out_values_size += t->nnz();
  }

  DenseTensor out_values = phi::Empty<T>(dev_ctx, {out_values_size});
  T* out_values_data = out_values.data<T>();
  DenseTensor out_cols = phi::Empty<int64_t>(dev_ctx, {out_values_size});
  int64_t* out_cols_data = out_cols.data<int64_t>();

  phi::DDim out_dims = phi::funcs::ComputeAndCheckShape(true, x_dims, axis);
  int x_dim = x_dims[0].size();
  // now csr only support 2-d and 3-d size
  if (x_dim == 2) {
    if (axis == 0) {
      ConcatCsr2D0A<T, Context>(out_crows_size,
                                num_split,
                                x,
                                dev_ctx,
                                out_values_data,
                                values_data_vec,
                                out_cols_data,
                                cols_data_vec,
                                crows_data_vec,
                                out,
                                &out_cols,
                                &out_values,
                                &out_dims);
    } else {
      ConcatCsr2D1A<T, Context>(x,
                                out_crows_size,
                                dev_ctx,
                                num_split,
                                crows_data_vec,
                                out_cols_data,
                                cols_data_vec,
                                out_values_data,
                                values_data_vec,
                                out,
                                &out_cols,
                                &out_values,
                                &out_dims);
    }
  } else if (x_dim == 3) {
    if (axis == 0) {
      ConcatCsr3D0A<T, Context>(num_split,
                                x,
                                out_crows_size,
                                dev_ctx,
                                out,
                                &out_values,
                                &out_cols,
                                &out_dims);

    } else if (axis == 1) {
      ConcatCsr3D1A<T, Context>(x,
                                out_crows_size,
                                num_split,
                                dev_ctx,
                                crows_data_vec,
                                values_data_vec,
                                cols_data_vec,
                                out_values_data,
                                out_cols_data,
                                out,
                                &out_cols,
                                &out_values,
                                &out_dims);
    } else {
      ConcatCsr3D2A<T, Context>(x,
                                dev_ctx,
                                num_split,
                                crows_data_vec,
                                out_cols_data,
                                cols_data_vec,
                                out_values_data,
                                values_data_vec,
                                out,
                                &out_cols,
                                &out_values,
                                &out_dims);
    }
  } else {
    // throw exception
    phi::errors::InvalidArgument(
        "Concat for Sparse CSR Tensor only support 2-D or 3-D, but got %d-D.",
        x_dim);
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(concat_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ConcatCooKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t) {}

PD_REGISTER_KERNEL(concat_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ConcatCsrKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int,
                   uint8_t,
                   int8_t,
                   int16_t) {}
