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

#pragma once
#include <unordered_set>
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/trunc_kernel.h"

namespace phi {
namespace sparse {

#define DEFINE_SPARSE_UNARY_KERNEL(prefix)                                 \
  template <typename T, typename Context>                                  \
  void prefix##CooKernel(const Context& dev_ctx,                           \
                         const SparseCooTensor& x,                         \
                         SparseCooTensor* out) {                           \
    EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);                       \
    phi::prefix##Kernel<T, Context>(                                       \
        dev_ctx, x.non_zero_elements(), out->mutable_non_zero_elements()); \
  }                                                                        \
                                                                           \
  template <typename T, typename Context>                                  \
  void prefix##CsrKernel(const Context& dev_ctx,                           \
                         const SparseCsrTensor& x,                         \
                         SparseCsrTensor* out) {                           \
    EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);                       \
    phi::prefix##Kernel<T, Context>(                                       \
        dev_ctx, x.non_zero_elements(), out->mutable_non_zero_elements()); \
  }

#define DEFINE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(prefix, attr)         \
  template <typename T, typename Context>                              \
  void prefix##CooKernel(const Context& dev_ctx,                       \
                         const SparseCooTensor& x,                     \
                         float attr,                                   \
                         SparseCooTensor* out) {                       \
    EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);                   \
    phi::prefix##Kernel<T, Context>(dev_ctx,                           \
                                    x.non_zero_elements(),             \
                                    attr,                              \
                                    out->mutable_non_zero_elements()); \
  }                                                                    \
                                                                       \
  template <typename T, typename Context>                              \
  void prefix##CsrKernel(const Context& dev_ctx,                       \
                         const SparseCsrTensor& x,                     \
                         float attr,                                   \
                         SparseCsrTensor* out) {                       \
    EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);                   \
    phi::prefix##Kernel<T, Context>(dev_ctx,                           \
                                    x.non_zero_elements(),             \
                                    attr,                              \
                                    out->mutable_non_zero_elements()); \
  }

DEFINE_SPARSE_UNARY_KERNEL(Sin)
DEFINE_SPARSE_UNARY_KERNEL(Tan)
DEFINE_SPARSE_UNARY_KERNEL(Asin)
DEFINE_SPARSE_UNARY_KERNEL(Atan)
DEFINE_SPARSE_UNARY_KERNEL(Sinh)
DEFINE_SPARSE_UNARY_KERNEL(Tanh)
DEFINE_SPARSE_UNARY_KERNEL(Asinh)
DEFINE_SPARSE_UNARY_KERNEL(Atanh)
DEFINE_SPARSE_UNARY_KERNEL(Sqrt)
DEFINE_SPARSE_UNARY_KERNEL(Square)
DEFINE_SPARSE_UNARY_KERNEL(Log1p)
DEFINE_SPARSE_UNARY_KERNEL(Relu)
DEFINE_SPARSE_UNARY_KERNEL(Abs)
DEFINE_SPARSE_UNARY_KERNEL(Expm1)
DEFINE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(Pow, factor)
DEFINE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(Relu6, threshold)
DEFINE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(LeakyRelu, alpha)

template <typename T, typename Context>
void ScaleCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    float scale,
                    float bias,
                    bool bias_after_scale,
                    SparseCooTensor* out) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);
  phi::ScaleKernel<T, Context>(dev_ctx,
                               x.non_zero_elements(),
                               scale,
                               bias,
                               bias_after_scale,
                               out->mutable_non_zero_elements());
}

template <typename T, typename Context>
void ScaleCsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    float scale,
                    float bias,
                    bool bias_after_scale,
                    SparseCsrTensor* out) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);
  phi::ScaleKernel<T, Context>(dev_ctx,
                               x.non_zero_elements(),
                               scale,
                               bias,
                               bias_after_scale,
                               out->mutable_non_zero_elements());
}

template <typename T, typename Context>
void CastCooKernel(const Context& dev_ctx,
                   const SparseCooTensor& x,
                   DataType index_dtype,
                   DataType value_dtype,
                   SparseCooTensor* out) {
  out->set_dims(x.dims());

  const DenseTensor& x_indices = x.indices();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_indices = out->mutable_indices();
  DenseTensor* out_values = out->mutable_non_zero_elements();

  if (index_dtype == DataType::UNDEFINED) {
    *out_indices = x_indices;
  } else {
    phi::MetaTensor meta(out_indices);
    meta.set_dims(x_indices.dims());
    meta.set_dtype(index_dtype);

    PD_VISIT_INTEGRAL_TYPES(x_indices.dtype(), "CastCooKernel", [&] {
      phi::CastKernel<data_t, Context>(
          dev_ctx, x_indices, index_dtype, out_indices);
    });
  }

  if (value_dtype == DataType::UNDEFINED) {
    phi::Copy(dev_ctx, x_values, dev_ctx.GetPlace(), false, out_values);
  } else {
    phi::MetaTensor meta(out_values);
    meta.set_dims(x_values.dims());
    meta.set_dtype(value_dtype);
    phi::CastKernel<T, Context>(dev_ctx, x_values, value_dtype, out_values);
  }
}

template <typename T, typename Context>
void CastCsrKernel(const Context& dev_ctx,
                   const SparseCsrTensor& x,
                   DataType index_dtype,
                   DataType value_dtype,
                   SparseCsrTensor* out) {
  out->set_dims(x.dims());

  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_cols = x.cols();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_crows = out->mutable_crows();
  DenseTensor* out_cols = out->mutable_cols();
  DenseTensor* out_values = out->mutable_non_zero_elements();

  if (index_dtype == DataType::UNDEFINED) {
    *out_crows = x_crows;
    *out_cols = x_cols;
  } else {
    phi::MetaTensor crows_meta(out_crows);
    crows_meta.set_dims(x_crows.dims());
    crows_meta.set_dtype(index_dtype);

    PD_VISIT_INTEGRAL_TYPES(x_crows.dtype(), "CastCsrKernel", [&] {
      phi::CastKernel<data_t, Context>(
          dev_ctx, x_crows, index_dtype, out_crows);
    });

    phi::MetaTensor cols_meta(out_cols);
    cols_meta.set_dims(x_cols.dims());
    cols_meta.set_dtype(index_dtype);

    PD_VISIT_INTEGRAL_TYPES(x_cols.dtype(), "CastCsrKernel", [&] {
      phi::CastKernel<data_t, Context>(dev_ctx, x_cols, index_dtype, out_cols);
    });
  }

  if (value_dtype == DataType::UNDEFINED) {
    phi::Copy(dev_ctx, x_values, dev_ctx.GetPlace(), false, out_values);
  } else {
    phi::MetaTensor meta(out_values);
    meta.set_dims(x_values.dims());
    meta.set_dtype(value_dtype);
    phi::CastKernel<T, Context>(dev_ctx, x_values, value_dtype, out_values);
  }
}

template <typename T, typename Context>
void TransposeCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const std::vector<int>& dims,
                        SparseCooTensor* out) {
  EmptyLikeCooKernel(dev_ctx, x, out);
  const DenseTensor& x_indices = x.indices();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_indices = out->mutable_indices();
  DenseTensor* out_values = out->mutable_non_zero_elements();

  const int64_t* x_indices_data = x_indices.data<int64_t>();
  int64_t* out_indices_data = out_indices->data<int64_t>();
  int64_t x_nnz = x.nnz();
  for (unsigned int i = 0; i < dims.size(); ++i) {
    for (int64_t j = 0; j < x_nnz; ++j) {
      out_indices_data[j + i * x_nnz] = x_indices_data[j + dims[i] * x_nnz];
    }
  }

  DDim out_dims(x.dims());
  out_dims.transpose(dims);
  phi::Copy(dev_ctx, x_values, dev_ctx.GetPlace(), false, out_values);
  out->Resize(out_dims, x.sparse_dim(), x_nnz);
}

template <typename T, typename Context>
void TransposeCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        const std::vector<int>& dims,
                        SparseCsrTensor* out) {
  unsigned int n_dim = dims.size();
  DDim out_dims(x.dims());
  EmptyLikeCooKernel(dev_ctx, x, out);
  out->Resize(out_dims, x.nnz());
  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_cols = x.cols();
  const DenseTensor& x_values = x.non_zero_elements();
  DenseTensor* out_crows = out->mutable_crows();
  DenseTensor* out_cols = out->mutable_cols();
  DenseTensor* out_values = out->mutable_non_zero_elements();

  // return a copy of x
  if (dims[0] == 0 && dims[1] == 1 && (n_dim == 2 || dims[2] == 2)) {
    *out_crows = x_crows;
    *out_cols = x_cols;
    phi::Copy(dev_ctx, x_values, dev_ctx.GetPlace(), false, out_values);
    return;
  }
  // transpose by two stages
  if (dims[0] == 1 && dims[1] == 2) {  // dims == {1, 2, 0}
    SparseCsrTensor* temp;
    TransposeCsrKernel<T>(dev_ctx, x, {1, 0, 2}, temp);
    TransposeCsrKernel<T>(dev_ctx, *temp, {0, 2, 1}, out);
    //  SparseCsrTensor(const DenseTensor& non_zero_crows,
    //                  const DenseTensor& non_zero_cols,
    //                  const DenseTensor& non_zero_elements,
    //                  const DDim& dims);
    //  DenseTensor temp_crows;
    //  DenseTensor temp_cols;
    //  DenseTensor temp_elements;
    return;
  } else if (dims[0] == 2 && dims[1] == 0) {  // dims == {2, 0, 1}
    SparseCsrTensor* temp;
    TransposeCsrKernel<T>(dev_ctx, x, {0, 2, 1}, temp);
    TransposeCsrKernel<T>(dev_ctx, *temp, {1, 0, 2}, out);
    return;
  } else if (dims[0] == 2 && dims[1] == 1) {  // dims == {2, 1, 0}
    SparseCsrTensor* temp;
    TransposeCsrKernel<T>(dev_ctx, x, {1, 0, 2}, temp);
    TransposeCsrKernel<T>(dev_ctx, *temp, {2, 0, 1}, out);
    return;
  }
  int* out_crows_data = out_crows->data<int>();
  int* out_cols_data = out_cols->data<int>();
  T* out_values_data = out_values->data<T>();
  const int* x_crows_data = x_crows.data<int>();
  const int* x_cols_data = x_cols.data<int>();
  const T* x_values_data = x_values.data<T>();

  if (n_dim == 2) {  // dims == {1, 0}
    // compute out_crows_data by x_cols_data
    for (int i = 0; i < out_dims[0]; ++i) {
      out_crows_data[i] = 0;
    }
    out_crows_data[out_dims[0]] = x.nnz();
    for (int i = 0; i < x.nnz(); ++i) {
      int j = x_cols_data[i];
      out_crows_data[j + 1]++;
    }
    for (int i = 1; i < out_dims[0]; ++i) {
      out_crows_data[i] += out_crows_data[i - 1];
    }
    // compute out_cols_data and out_values_data by out_crows_data and x
    std::unordered_set<int> cols_ptr;
    for (int i = 0; i < x.dims()[0]; ++i) {
      int start = x_crows_data[i];
      int end = x_crows_data[i + 1];
      for (int j = start; j < end; ++j) {
        int jj = x_cols_data[j];
        int jjj = out_crows_data[jj];
        int jjj_ptr = jjj + cols_ptr.count(jjj);
        out_cols_data[jjj_ptr] = i;
        out_values_data[jjj_ptr] = x_values_data[j];
        cols_ptr.insert(jjj);
      }
    }
  } else {  // n_dim == 3
    for (int k = 0; k < out_dims[0]; ++k) {
      if (dims[0] == 0) {  // dims == {0, 2, 1}
        int out_n_rows = out_dims[1];
        // compute out_crows_data by x_cols_data
        for (int i = 0; i < out_n_rows; ++i) {
          out_crows_data[i] = 0;
        }
        out_crows_data[out_n_rows] = x_crows_data[x.dims()[1]];
        for (int i = 0; i < out_crows_data[out_n_rows]; ++i) {
          int j = x_cols_data[i];
          out_crows_data[j + 1]++;
        }
        for (int i = 1; i < out_n_rows; ++i) {
          out_crows_data[i] += out_crows_data[i - 1];
        }
        // compute out_cols_data and out_values_data by out_crows_data and x
        std::unordered_set<int> cols_ptr;
        for (int i = 0; i < x.dims()[1]; ++i) {
          int start = x_crows_data[i];
          int end = x_crows_data[i + 1];
          for (int j = start; j < end; ++j) {
            int jj = x_cols_data[j];
            int jjj = out_crows_data[jj];
            int jjj_ptr = jjj + cols_ptr.count(jjj);
            out_cols_data[jjj_ptr] = i;
            out_values_data[jjj_ptr] = x_values_data[j];
            cols_ptr.insert(jjj);
          }
        }
        // x offset
        x_crows_data += x.dims()[1] + 1;
        x_cols_data += x_crows_data[x.dims()[1]];
        x_values_data += x_crows_data[x.dims()[1]];
      } else if (dims[0] == 1 && dims[1] == 0) {  // dims == {1, 0, 2}
        int out_n_rows = out_dims[1];
        int x_cols_offset = 0;
        for (int i = 0; i < x.dims()[0]; ++i) {
          int x_crows_index = i * (x.dims()[1] + 1);
          int start = x_crows_data[x_crows_index];
          int end = x_crows_data[x_crows_index + 1];
          out_crows_data[i + 1] = end - start;
          for (int j = start; j < end; ++j) {
            out_cols_data[j - start] = x_cols_data[x_cols_offset + j];
            out_values_data[j - start] = x_values_data[x_cols_offset + j];
          }
          x_cols_offset += x_crows_data[x_crows_index + x.dims()[1]];
        }
        for (int i = 1; i <= out_n_rows; ++i) {
          out_crows_data[i] += out_crows_data[i - 1];
        }
        // x offset
        x_crows_data += 1;
      }
      // out offset
      out_crows_data += out_dims[1] + 1;
      out_cols_data += x_crows_data[out_dims[1]];
      out_values_data += x_crows_data[out_dims[1]];
    }
  }
}

}  // namespace sparse
}  // namespace phi
