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
        phi::errors::InvalidArgument(                                          \
            "The input tensor X's shape "                                      \
            "should be identical with Y's shape. But received X's "            \
            "shape = [%s], Y's shape = [%s].",                                 \
            x.dims(),                                                          \
            y.dims()));                                                        \
    const auto& n_batch = x.dims().size() == 3 ? x.dims()[0] : 1;              \
    const auto& n_row = x.dims().size() == 2 ? x.dims()[0] : x.dims()[1];      \
    const auto& n_col = x.dims().size() == 2 ? x.dims()[1] : x.dims()[2];      \
    const auto& x_nnz = x.non_zero_elements().numel();                         \
    const auto* x_crows_data = x.non_zero_crows().data<int64_t>();             \
    const auto* x_cols_data = x.non_zero_cols().data<int64_t>();               \
    const auto* x_values_data = x.non_zero_elements().data<T>();               \
    const auto& y_nnz = y.non_zero_elements().numel();                         \
    const auto* y_crows_data = y.non_zero_crows().data<int64_t>();             \
    const auto* y_cols_data = y.non_zero_cols().data<int64_t>();               \
    const auto* y_values_data = y.non_zero_elements().data<T>();               \
    const auto func = funcs::name##Functor<T>();                               \
                                                                               \
    std::vector<int64_t> next(n_col, -1);                                      \
    std::vector<T> A_row(n_col, 0);                                            \
    std::vector<T> B_row(n_col, 0);                                            \
    int64_t nnz = 0;                                                           \
    std::vector<int64_t> out_crows_vec;                                        \
    std::vector<int64_t> out_cols_vec;                                         \
    std::vector<T> out_values_vec;                                             \
    std::vector<int64_t> out_batch_crows_vec;                                  \
    std::vector<int64_t> out_batch_cols_vec;                                   \
    std::vector<T> out_batch_values_vec;                                       \
    out_batch_crows_vec.reserve(x_nnz + y_nnz);                                \
    out_batch_cols_vec.reserve(x_nnz + y_nnz);                                 \
    out_batch_values_vec.reserve(x_nnz + y_nnz);                               \
    out_batch_crows_vec.push_back(0);                                          \
    int64_t x_prev_batch_nnz = 0;                                              \
    int64_t y_prev_batch_nnz = 0;                                              \
    for (int b = 0; b < n_batch; b++) {                                        \
      for (int64_t i = 0; i < n_row; i++) {                                    \
        int64_t head = -2;                                                     \
        int64_t length = 0;                                                    \
        if (i == 0) {                                                          \
          x_prev_batch_nnz +=                                                  \
              b == 0 ? 0 : x_crows_data[i + b * (n_row + 1) - 1];              \
        }                                                                      \
        int64_t i_start = x_crows_data[i + b * (n_row + 1)];                   \
        int64_t i_end = x_crows_data[i + b * (n_row + 1) + 1];                 \
        for (int64_t jj = i_start + x_prev_batch_nnz;                          \
             jj < i_end + x_prev_batch_nnz;                                    \
             jj++) {                                                           \
          int64_t j = x_cols_data[jj];                                         \
          A_row[j] += x_values_data[jj];                                       \
          if (next[j] == -1) {                                                 \
            next[j] = head;                                                    \
            head = j;                                                          \
            length++;                                                          \
          }                                                                    \
        }                                                                      \
        if (i == 0) {                                                          \
          y_prev_batch_nnz +=                                                  \
              b == 0 ? 0 : y_crows_data[i + b * (n_row + 1) - 1];              \
        }                                                                      \
        i_start = y_crows_data[i + b * (n_row + 1)];                           \
        i_end = y_crows_data[i + b * (n_row + 1) + 1];                         \
        for (int64_t jj = i_start + y_prev_batch_nnz;                          \
             jj < i_end + y_prev_batch_nnz;                                    \
             jj++) {                                                           \
          int64_t j = y_cols_data[jj];                                         \
          B_row[j] += y_values_data[jj];                                       \
          if (next[j] == -1) {                                                 \
            next[j] = head;                                                    \
            head = j;                                                          \
            length++;                                                          \
          }                                                                    \
        }                                                                      \
        for (int64_t jj = 0; jj < length; jj++) {                              \
          auto result = func(A_row[head], B_row[head]);                        \
          if (result != 0) {                                                   \
            out_batch_cols_vec.resize(nnz + 1);                                \
            out_batch_cols_vec[nnz] = head;                                    \
            out_batch_values_vec.resize(nnz + 1);                              \
            out_batch_values_vec[nnz] = result;                                \
            nnz++;                                                             \
          }                                                                    \
          int64_t tmp = head;                                                  \
          head = next[head];                                                   \
          next[tmp] = -1;                                                      \
          A_row[tmp] = 0;                                                      \
          B_row[tmp] = 0;                                                      \
        }                                                                      \
        out_batch_crows_vec.push_back(nnz);                                    \
      }                                                                        \
      nnz = 0;                                                                 \
      out_cols_vec.insert(out_cols_vec.end(),                                  \
                          out_batch_cols_vec.begin(),                          \
                          out_batch_cols_vec.end());                           \
      out_crows_vec.insert(out_crows_vec.end(),                                \
                           out_batch_crows_vec.begin(),                        \
                           out_batch_crows_vec.end());                         \
      out_values_vec.insert(out_values_vec.end(),                              \
                            out_batch_values_vec.begin(),                      \
                            out_batch_values_vec.end());                       \
      out_crows_vec.push_back(0);                                              \
      out_batch_cols_vec.clear();                                              \
      out_batch_crows_vec.clear();                                             \
      out_batch_values_vec.clear();                                            \
    }                                                                          \
    out_crows_vec.resize(out_crows_vec.size() - 1);                            \
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
    phi::DenseTensor out_crows = phi::Empty(dev_ctx, std::move(crows_meta));   \
    phi::DenseTensor out_cols = phi::Empty(dev_ctx, std::move(cols_meta));     \
    phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta)); \
    std::memcpy(out_crows.template data<int64_t>(),                            \
                out_crows_vec.data(),                                          \
                sizeof(int64_t) * out_crows_vec.size());                       \
    std::memcpy(out_cols.template data<int64_t>(),                             \
                out_cols_vec.data(),                                           \
                sizeof(int64_t) * out_cols_vec.size());                        \
    std::memcpy(out_values.template data<T>(),                                 \
                out_values_vec.data(),                                         \
                sizeof(T) * out_values_vec.size());                            \
    out->SetMember(out_crows, out_cols, out_values, x.dims());                 \
  }

#define DEFINE_COO_ELEMENTWISE_KERNEL(name)                           \
  template <typename T, typename Context>                             \
  void ElementWise##name##CooKernel(const Context& dev_ctx,           \
                                    const SparseCooTensor& x,         \
                                    const SparseCooTensor& y,         \
                                    SparseCooTensor* out) {           \
    const auto csr_x = SparseCooToCsr<T>(dev_ctx, x);                 \
    const auto csr_y = SparseCooToCsr<T>(dev_ctx, y);                 \
    DenseTensor non_zero_crows;                                       \
    DenseTensor non_zero_cols;                                        \
    DenseTensor non_zero_elements;                                    \
    SparseCsrTensor csr_out(                                          \
        non_zero_crows, non_zero_cols, non_zero_elements, x.dims());  \
    ElementWise##name##CsrKernel<T>(dev_ctx, csr_x, csr_y, &csr_out); \
    *out = SparseCsrToCoo<T>(dev_ctx, csr_out);                       \
  }

template <typename T, typename Context>
void ElementWiseDivideCsrKernel(const Context& dev_ctx,
                                const SparseCsrTensor& x,
                                const SparseCsrTensor& y,
                                SparseCsrTensor* out) {
  PADDLE_ENFORCE_EQ(x.dims(),
                    y.dims(),
                    phi::errors::InvalidArgument(
                        "The input tensor X's shape "
                        "should be identical with Y's shape. But received X's "
                        "shape = [%s], Y's shape = [%s].",
                        x.dims(),
                        y.dims()));
  const auto& n_batch = x.dims().size() == 3 ? x.dims()[0] : 1;
  const auto& n_row = x.dims().size() == 2 ? x.dims()[0] : x.dims()[1];
  const auto& n_col = x.dims().size() == 2 ? x.dims()[1] : x.dims()[2];
  const auto& x_nnz = x.non_zero_elements().numel();
  const auto& y_nnz = y.non_zero_elements().numel();
  const auto* y_crows_data = y.non_zero_crows().data<int64_t>();
  const auto* y_cols_data = y.non_zero_cols().data<int64_t>();
  const auto* y_values_data = y.non_zero_elements().data<T>();
  const auto func = funcs::DivideFunctor<T>();
  std::vector<int64_t> x_full_crows;
  x_full_crows.reserve(n_batch * (n_row + 1));
  for (int b = 0; b < n_batch; ++b) {
    for (int64_t i = 0; i < n_row + 1; ++i) {
      x_full_crows.push_back(n_col * i);
    }
  }
  std::vector<int64_t> x_full_cols;
  x_full_cols.reserve(n_batch * n_col * n_row);
  for (int b = 0; b < n_batch; ++b) {
    for (int64_t i = 0; i < n_row; ++i) {
      for (int64_t j = 0; j < n_col; ++j) {
        x_full_cols.push_back(j);
      }
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
  std::vector<int64_t> out_batch_crows_vec;
  std::vector<int64_t> out_batch_cols_vec;
  std::vector<T> out_batch_values_vec;
  out_batch_crows_vec.reserve(x_nnz + y_nnz);
  out_batch_cols_vec.reserve(x_nnz + y_nnz);
  out_batch_values_vec.reserve(x_nnz + y_nnz);
  out_batch_crows_vec.push_back(0);
  int64_t x_prev_batch_nnz = 0;
  int64_t y_prev_batch_nnz = 0;
  for (int b = 0; b < n_batch; b++) {
    for (int64_t i = 0; i < n_row; i++) {
      int64_t head = -2;
      int64_t length = 0;
      if (i == 0) {
        x_prev_batch_nnz += b == 0 ? 0 : x_crows_data[i + b * (n_row + 1) - 1];
      }
      int64_t i_start = x_crows_data[i + b * (n_row + 1)];
      int64_t i_end = x_crows_data[i + b * (n_row + 1) + 1];
      for (int64_t jj = i_start + x_prev_batch_nnz;
           jj < i_end + x_prev_batch_nnz;
           jj++) {
        int64_t j = x_cols_data[jj];
        A_row[j] += x_values_data[jj];
        if (next[j] == -1) {
          next[j] = head;
          head = j;
          length++;
        }
      }
      if (i == 0) {
        y_prev_batch_nnz += b == 0 ? 0 : y_crows_data[i + b * (n_row + 1) - 1];
      }
      i_start = y_crows_data[i + b * (n_row + 1)];
      i_end = y_crows_data[i + b * (n_row + 1) + 1];
      for (int64_t jj = i_start + y_prev_batch_nnz;
           jj < i_end + y_prev_batch_nnz;
           jj++) {
        int64_t j = y_cols_data[jj];
        B_row[j] += y_values_data[jj];
        if (next[j] == -1) {
          next[j] = head;
          head = j;
          length++;
        }
      }
      for (int64_t jj = 0; jj < length; jj++) {
        auto result = func(A_row[head], B_row[head]);
        if (result != 0) {
          out_batch_cols_vec.resize(nnz + 1);
          out_batch_cols_vec[nnz] = head;
          out_batch_values_vec.resize(nnz + 1);
          out_batch_values_vec[nnz] = result;
          nnz++;
        }
        int64_t tmp = head;
        head = next[head];
        next[tmp] = -1;
        A_row[tmp] = 0;
        B_row[tmp] = 0;
      }
      out_batch_crows_vec.push_back(nnz);
    }
    nnz = 0;
    out_cols_vec.insert(out_cols_vec.end(),
                        out_batch_cols_vec.begin(),
                        out_batch_cols_vec.end());
    out_crows_vec.insert(out_crows_vec.end(),
                         out_batch_crows_vec.begin(),
                         out_batch_crows_vec.end());
    out_values_vec.insert(out_values_vec.end(),
                          out_batch_values_vec.begin(),
                          out_batch_values_vec.end());
    out_crows_vec.push_back(0);
    out_batch_cols_vec.clear();
    out_batch_crows_vec.clear();
    out_batch_values_vec.clear();
  }
  out_crows_vec.resize(out_crows_vec.size() - 1);
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
  std::memcpy(out_crows.template data<int64_t>(),
              out_crows_vec.data(),
              sizeof(int64_t) * out_crows_vec.size());
  std::memcpy(out_cols.template data<int64_t>(),
              out_cols_vec.data(),
              sizeof(int64_t) * out_cols_vec.size());
  std::memcpy(out_values.template data<T>(),
              out_values_vec.data(),
              sizeof(T) * out_values_vec.size());
  out->SetMember(out_crows, out_cols, out_values, x.dims());
}

DEFINE_CSR_ELEMENTWISE_KERNEL(Add)
DEFINE_CSR_ELEMENTWISE_KERNEL(Subtract)
DEFINE_CSR_ELEMENTWISE_KERNEL(Multiply)

DEFINE_COO_ELEMENTWISE_KERNEL(Add)
DEFINE_COO_ELEMENTWISE_KERNEL(Subtract)
DEFINE_COO_ELEMENTWISE_KERNEL(Multiply)
DEFINE_COO_ELEMENTWISE_KERNEL(Divide)

}  // namespace sparse
}  // namespace phi

// sparse_elementwise_add
PD_REGISTER_KERNEL(sparse_elementwise_add_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_elementwise_add_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseAddCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(sparse_elementwise_sub_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_elementwise_sub_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseSubtractCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(sparse_elementwise_mul_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_elementwise_mul_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseMultiplyCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(sparse_elementwise_div_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseDivideCsrKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_elementwise_div_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ElementWiseDivideCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
