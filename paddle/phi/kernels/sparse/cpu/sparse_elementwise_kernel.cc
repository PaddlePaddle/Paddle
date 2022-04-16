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

#include "paddle/phi/kernels/sparse/sparse_elementwise_kernel.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

#define DEFINE_CSR_ELEMENTWISE_KERNEL(name)                                    \
  template <typename T, typename Context>                                      \
  void ElementWise##name##CsrKernel(const Context& dev_ctx,                    \
                                    const SparseCsrTensor& x,                  \
                                    const SparseCsrTensor& y,                  \
                                    SparseCsrTensor* out) {                    \
    PADDLE_ENFORCE_EQ(                                                         \
        x.dims(),                                                              \
        y.dims(),                                                              \
        "ValueError: Mismatched shape. Expected x.dims = y.dims, "             \
        "but got x.dims = %s, y.dims = %s",                                    \
        x.dims().to_str(),                                                     \
        y.dims().to_str());                                                    \
    const auto& n_row = x.dims()[0];                                           \
    const auto& n_col = x.dims()[1];                                           \
    const auto& x_nnz = x.non_zero_elements().numel();                         \
    const auto* x_crows_data = x.non_zero_crows().data<int64_t>();             \
    const auto* x_cols_data = x.non_zero_cols().data<int64_t>();               \
    const auto* x_values_data = x.non_zero_elements().data<T>();               \
    const auto& y_nnz = y.non_zero_elements().numel();                         \
    const auto* y_crows_data = y.non_zero_crows().data<int64_t>();             \
    const auto* y_cols_data = y.non_zero_cols().data<int64_t>();               \
    const auto* y_values_data = y.non_zero_elements().data<T>();               \
    const auto place = dev_ctx.GetPlace();                                     \
    const auto func = funcs::name##Functor<T>();                               \
                                                                               \
    std::vector<int64_t> next(n_col, -1);                                      \
    std::vector<T> A_row(n_col, 0);                                            \
    std::vector<T> B_row(n_col, 0);                                            \
    int64_t nnz = 0;                                                           \
                                                                               \
    std::vector<int64_t> out_crows_vec;                                        \
    std::vector<int64_t> out_cols_vec;                                         \
    std::vector<T> out_values_vec;                                             \
    out_crows_vec.reserve(x_nnz + y_nnz);                                      \
    out_cols_vec.reserve(x_nnz + y_nnz);                                       \
    out_values_vec.reserve(x_nnz + y_nnz);                                     \
                                                                               \
    out_crows_vec.push_back(0);                                                \
    for (int64_t i = 0; i < n_row; i++) {                                      \
      int64_t head = -2;                                                       \
      int64_t length = 0;                                                      \
      int64_t i_start = x_crows_data[i];                                       \
      int64_t i_end = x_crows_data[i + 1];                                     \
      for (int64_t jj = i_start; jj < i_end; jj++) {                           \
        int64_t j = x_cols_data[jj];                                           \
        A_row[j] += x_values_data[jj];                                         \
        if (next[j] == -1) {                                                   \
          next[j] = head;                                                      \
          head = j;                                                            \
          length++;                                                            \
        }                                                                      \
      }                                                                        \
      i_start = y_crows_data[i];                                               \
      i_end = y_crows_data[i + 1];                                             \
      for (int64_t jj = i_start; jj < i_end; jj++) {                           \
        int64_t j = y_cols_data[jj];                                           \
        B_row[j] += y_values_data[jj];                                         \
        if (next[j] == -1) {                                                   \
          next[j] = head;                                                      \
          head = j;                                                            \
          length++;                                                            \
        }                                                                      \
      }                                                                        \
      for (int64_t jj = 0; jj < length; jj++) {                                \
        auto result = func(A_row[head], B_row[head]);                          \
        if (result != 0) {                                                     \
          out_cols_vec.resize(nnz + 1);                                        \
          out_cols_vec[nnz] = head;                                            \
          out_values_vec.resize(nnz + 1);                                      \
          out_values_vec[nnz] = result;                                        \
          nnz++;                                                               \
        }                                                                      \
        int64_t tmp = head;                                                    \
        head = next[head];                                                     \
        next[tmp] = -1;                                                        \
        A_row[tmp] = 0;                                                        \
        B_row[tmp] = 0;                                                        \
      }                                                                        \
      out_crows_vec.push_back(nnz);                                            \
    }                                                                          \
                                                                               \
    DenseTensorMeta crows_meta(                                                \
        DataType::INT64,                                                       \
        phi::make_ddim({static_cast<int64_t>(out_crows_vec.size())}),          \
        DataLayout::NCHW);                                                     \
    DenseTensorMeta cols_meta(                                                 \
        DataType::INT64,                                                       \
        phi::make_ddim({static_cast<int64_t>(out_cols_vec.size())}),           \
        DataLayout::NCHW);                                                     \
    DenseTensorMeta values_meta(                                               \
        paddle::experimental::CppTypeToDataType<T>::Type(),                    \
        phi::make_ddim({static_cast<int64_t>(out_values_vec.size())}),         \
        DataLayout::NCHW);                                                     \
                                                                               \
    phi::DenseTensor out_crows = phi::Empty(dev_ctx, std::move(crows_meta));   \
    phi::DenseTensor out_cols = phi::Empty(dev_ctx, std::move(cols_meta));     \
    phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta)); \
                                                                               \
    std::memcpy(out_crows.mutable_data<int64_t>(place),                        \
                out_crows_vec.data(),                                          \
                sizeof(int64_t) * out_crows_vec.size());                       \
    std::memcpy(out_cols.mutable_data<int64_t>(place),                         \
                out_cols_vec.data(),                                           \
                sizeof(int64_t) * out_cols_vec.size());                        \
    std::memcpy(out_values.mutable_data<T>(place),                             \
                out_values_vec.data(),                                         \
                sizeof(T) * out_values_vec.size());                            \
                                                                               \
    out->SetMember(out_crows, out_cols, out_values, x.dims());                 \
  }

DEFINE_CSR_ELEMENTWISE_KERNEL(Add)

DEFINE_CSR_ELEMENTWISE_KERNEL(Subtract)

DEFINE_CSR_ELEMENTWISE_KERNEL(Multiply)

template <typename T, typename Context>
void ElementWiseDivideCsrKernel(const Context& dev_ctx,
                                const SparseCsrTensor& x,
                                const SparseCsrTensor& y,
                                SparseCsrTensor* out) {
  const auto& n_row = x.dims()[0];
  const auto& n_col = x.dims()[1];
  const auto& x_nnz = x.non_zero_elements().numel();
  const auto* y_crows_data = y.non_zero_crows().data<int64_t>();
  const auto* y_cols_data = y.non_zero_cols().data<int64_t>();
  const auto* y_values_data = y.non_zero_elements().data<T>();
  const auto& y_nnz = y.non_zero_elements().numel();
  const auto place = dev_ctx.GetPlace();
  const auto func = funcs::DivideFunctor<T>();

  std::vector<int64_t> x_full_crows;
  x_full_crows.reserve(n_col);
  for (int64_t i = 0; i < n_col; ++i) {
    x_full_crows.push_back(n_col * i);
  }

  std::vector<int64_t> x_full_cols;
  x_full_cols.reserve(n_col * n_row);
  for (int64_t i = 0; i < n_row; ++i) {
    for (int64_t j = 0; j < n_col; ++j) {
      x_full_cols.push_back(j);
    }
  }

  const auto* x_crows_data = x_full_crows.data();
  const auto* x_cols_data = x_full_cols.data();
  const auto* x_values_data =
      SparseCsrToDense<T>(dev_ctx, x).template data<T>();

  std::vector<int64_t> next(n_col, -1);
  std::vector<T> A_row(n_col, 0);
  std::vector<T> B_row(n_col, 0);
  int64_t nnz = 0;

  std::vector<int64_t> out_crows_vec;
  std::vector<int64_t> out_cols_vec;
  std::vector<T> out_values_vec;
  out_crows_vec.reserve(x_nnz + y_nnz);
  out_cols_vec.reserve(x_nnz + y_nnz);
  out_values_vec.reserve(x_nnz + y_nnz);

  out_crows_vec.push_back(0);

  for (int64_t i = 0; i < n_row; i++) {
    int64_t head = -2;
    int64_t length = 0;
    int64_t i_start = x_crows_data[i];
    int64_t i_end = x_crows_data[i + 1];
    for (int64_t jj = i_start; jj < i_end; jj++) {
      int64_t j = x_cols_data[jj];
      A_row[j] += x_values_data[jj];
      if (next[j] == -1) {
        next[j] = head;
        head = j;
        length++;
      }
    }
    i_start = y_crows_data[i];
    i_end = y_crows_data[i + 1];
    for (int64_t jj = i_start; jj < i_end; jj++) {
      int64_t j = y_cols_data[jj];
      B_row[j] += y_values_data[jj];
      if (next[j] == -1) {
        next[j] = head;
        head = j;
        length++;
      }
    }

    //    for (int64_t jj = 0; jj < n_col; jj++) {
    //      B_row[jj] = y_values_data[i*n_col+jj];
    //    }

    std::vector<float> x_dense_data = {
        0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 3.2, 0.0, 0.0, 3.2, 0.0, 0.0};
    std::vector<float> y_dense_data = {
        0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.5, 0.7, 0.0, 3.5, 0.7};

    for (int64_t jj = 0; jj < length; jj++) {
      auto result = func(A_row[head], B_row[head]);
      if (result != 0) {
        out_cols_vec.resize(nnz + 1);
        out_cols_vec[nnz] = head;
        out_values_vec.resize(nnz + 1);
        out_values_vec[nnz] = result;
        nnz++;
      }
      int64_t tmp = head;
      head = next[head];
      next[tmp] = -1;
      A_row[tmp] = 0;
      B_row[tmp] = 0;
    }
    out_crows_vec.push_back(nnz);
  }

  DenseTensorMeta crows_meta(
      DataType::INT64,
      phi::make_ddim({static_cast<int64_t>(out_crows_vec.size())}),
      DataLayout::NCHW);
  DenseTensorMeta cols_meta(
      DataType::INT64,
      phi::make_ddim({static_cast<int64_t>(out_cols_vec.size())}),
      DataLayout::NCHW);
  DenseTensorMeta values_meta(
      paddle::experimental::CppTypeToDataType<T>::Type(),
      phi::make_ddim({static_cast<int64_t>(out_values_vec.size())}),
      DataLayout::NCHW);

  phi::DenseTensor out_crows = phi::Empty(dev_ctx, std::move(crows_meta));
  phi::DenseTensor out_cols = phi::Empty(dev_ctx, std::move(cols_meta));
  phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));

  std::memcpy(out_crows.mutable_data<int64_t>(place),
              out_crows_vec.data(),
              sizeof(int64_t) * out_crows_vec.size());
  std::memcpy(out_cols.mutable_data<int64_t>(place),
              out_cols_vec.data(),
              sizeof(int64_t) * out_cols_vec.size());
  std::memcpy(out_values.mutable_data<T>(place),
              out_values_vec.data(),
              sizeof(T) * out_values_vec.size());

  out->SetMember(out_crows, out_cols, out_values, x.dims());
}

template <typename T, typename Context>
void ElementWiseDivideCsrKernelnew(const Context& dev_ctx,
                                   const SparseCsrTensor& x,
                                   const SparseCsrTensor& y,
                                   SparseCsrTensor* out) {
  auto denseX = SparseCsrToDense<T>(dev_ctx, x);
  auto denseY = SparseCsrToDense<T>(dev_ctx, y);
  auto denseOut = phi::Divide<T>(dev_ctx, denseX, denseY);

  *out = DenseToSparseCsr<T>(dev_ctx, denseOut);
}

}  // namespace sparse
}  // namespace phi

// sparse_elementwise_add
PD_REGISTER_KERNEL(sparse_elementwise_add,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_elementwise_sub,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_elementwise_mul,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_elementwise_div,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseDivideCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
