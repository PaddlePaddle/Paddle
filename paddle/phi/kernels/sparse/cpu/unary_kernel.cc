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
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_grad_kernel_impl.h"
#include "paddle/phi/kernels/sparse/impl/unary_kernel_impl.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void DivCooScalarKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        float scalar,
                        SparseCooTensor* out) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, out);

  auto eigen_out =
      phi::EigenVector<T>::Flatten(*(out->mutable_non_zero_elements()));
  auto eigen_x = phi::EigenVector<T>::Flatten(x.non_zero_elements());
  auto& dev = *dev_ctx.eigen_device();

  phi::funcs::EigenDiv<std::decay_t<decltype(dev)>, T>::Eval(
      dev, eigen_out, eigen_x, static_cast<T>(scalar));
}

template <typename T, typename Context>
void DivCsrScalarKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        float scalar,
                        SparseCsrTensor* out) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, out);

  auto eigen_out =
      phi::EigenVector<T>::Flatten(*(out->mutable_non_zero_elements()));
  auto eigen_x = phi::EigenVector<T>::Flatten(x.non_zero_elements());
  auto& dev = *dev_ctx.eigen_device();

  phi::funcs::EigenDiv<std::decay_t<decltype(dev)>, T>::Eval(
      dev, eigen_out, eigen_x, static_cast<T>(scalar));
}

template <typename T, typename Context>
void TransposeCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const std::vector<int>& perm,
                        SparseCooTensor* out) {
  // create out sparse tensor
  int64_t x_nnz = x.nnz();
  DDim out_dims = x.dims().transpose(perm);
  DenseTensor out_indices = EmptyLike<int64_t, Context>(dev_ctx, x.indices());
  DenseTensor out_values(x.values());
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
  // create out sparse tensor
  DDim out_dims = x.dims().transpose(perm);
  DenseTensor out_crows;
  if (n_dim == 2) {
    out_crows = Empty<int64_t, Context>(dev_ctx, {out_dims[0] + 1});
  } else {
    out_crows =
        Empty<int64_t, Context>(dev_ctx, {out_dims[0] * (out_dims[1] + 1)});
  }
  DenseTensor out_cols = EmptyLike<int64_t, Context>(dev_ctx, x.cols());
  DenseTensor out_values = EmptyLike<T, Context>(dev_ctx, x.values());
  out->SetMember(out_crows, out_cols, out_values, out_dims);

  const DenseTensor& x_crows = x.crows();
  const DenseTensor& x_cols = x.cols();
  const DenseTensor& x_values = x.non_zero_elements();

  // return a copy of x
  if (perm[0] == 0 && perm[1] == 1 && (n_dim == 2 || perm[2] == 2)) {
    phi::Copy(dev_ctx, x_crows, dev_ctx.GetPlace(), false, &out_crows);
    phi::Copy(dev_ctx, x_cols, dev_ctx.GetPlace(), false, &out_cols);
    phi::Copy(dev_ctx, x_values, dev_ctx.GetPlace(), false, &out_values);
    return;
  }
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
      int j = x_cols_data[i];
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
    int out_n_rows = out_dims[1];
    int x_n_rows = x.dims()[1];
    for (int k = 0; k < out_dims[0]; ++k) {
      if (perm[0] == 0) {  // perm == {0, 2, 1}
        // compute out_crows_data by x_cols_data
        for (int i = 0; i < out_n_rows; ++i) {
          out_crows_data[i] = 0;
        }
        for (int i = 0; i < x_crows_data[x_n_rows]; ++i) {
          int j = x_cols_data[i];
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
        int x_cols_offset = 0;
        int out_cols_index = 0;
        for (int i = 0; i < x.dims()[0]; ++i) {
          int x_crows_index = i * (x_n_rows + 1);
          int start = x_crows_data[x_crows_index + k];
          int end = x_crows_data[x_crows_index + 1 + k];
          out_crows_data[i + 1] = end - start;
          for (int j = start; j < end; ++j) {
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
}  // namespace sparse
}  // namespace phi

#define PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(name, prefix)          \
  PD_REGISTER_KERNEL(name##_coo,                                   \
                     CPU,                                          \
                     ALL_LAYOUT,                                   \
                     phi::sparse::prefix##CooKernel,               \
                     float,                                        \
                     double) {                                     \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO); \
  }                                                                \
                                                                   \
  PD_REGISTER_KERNEL(name##_csr,                                   \
                     CPU,                                          \
                     ALL_LAYOUT,                                   \
                     phi::sparse::prefix##CsrKernel,               \
                     float,                                        \
                     double) {                                     \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR); \
  }

PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(sin, Sin)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(tan, Tan)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(asin, Asin)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(atan, Atan)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(sinh, Sinh)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(tanh, Tanh)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(asinh, Asinh)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(atanh, Atanh)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(sqrt, Sqrt)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(square, Square)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(log1p, Log1p)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(relu, Relu)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(abs, Abs)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(pow, Pow)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(scale, Scale)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(expm1, Expm1)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(relu6, Relu6)
PD_REGISTER_SPARSE_UNARY_CPU_KERNEL(leaky_relu, LeakyRelu)

PD_REGISTER_KERNEL(divide_coo_scalar,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::DivCooScalarKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(divide_csr_scalar,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::DivCsrScalarKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(cast_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CastCooKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(cast_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CastCsrKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

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
