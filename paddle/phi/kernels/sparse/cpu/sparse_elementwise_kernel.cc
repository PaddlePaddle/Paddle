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

#include "paddle/phi/kernels/funcs/elementwise_functor.h"
//#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
//#include "paddle/phi/kernels/funcs/sparse/sparse_elementwise_base.h"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

#define CSR_ELEMENTWISE_API_NAME(name) sparse_elementwise_##name

#define DEFINE_CSR_ELEMENTWISE_KERNEL(name)                                    \
  template <typename T, typename Context>                                      \
  void ElementWise##name##CsrKernel(const Context& dev_ctx,                    \
                                    const SparseCsrTensor& x,                  \
                                    const SparseCsrTensor& y,                  \
                                    SparseCsrTensor* out) {                    \
    PADDLE_ENFORCE_EQ(                                                         \
        x.dims(),                                                              \
        y.dims(),                                                              \
        "ValueError: Mismatched shape. Expected x.dim = y.dim.");      \
    const DDim& x_dims = x.dims();                                             \
    const auto& n_row = x_dims[0];                                             \
    const auto& n_col = x_dims[1];                                             \
    const auto& x_crows = x.non_zero_crows();                                  \
    const auto& x_cols = x.non_zero_cols();                                    \
    const auto& x_values = x.non_zero_elements();                              \
    const auto& x_nnz = x.non_zero_elements().numel();                         \
    const auto* x_crows_data = x_crows.data<int64_t>();                        \
    const auto* x_cols_data = x_cols.data<int64_t>();                          \
    const auto* x_values_data = x_values.data<T>();                            \
    const auto& y_crows = y.non_zero_crows();                                  \
    const auto& y_cols = y.non_zero_cols();                                    \
    const auto& y_values = y.non_zero_elements();                              \
    const auto& y_nnz = y.non_zero_elements().numel();                         \
    const auto* y_crows_data = y_crows.data<int64_t>();                        \
    const auto* y_cols_data = y_cols.data<int64_t>();                          \
    const auto* y_values_data = y_values.data<T>();                            \
    const auto place = dev_ctx.GetPlace();                                     \
    const auto func = funcs::name##Functor<T>();                               \
    std::vector<int64_t> next(n_col, -1);                                      \
    std::vector<T> A_row(n_col, 0);                                            \
    std::vector<T> B_row(n_col, 0);                                            \
    int64_t nnz = 0;                                                           \
    std::vector<int64_t> Cp;                                                   \
    Cp.reserve(x_nnz + y_nnz);                                                 \
    std::vector<int64_t> Cj;                                                   \
    Cj.reserve(x_nnz + y_nnz);                                                 \
    std::vector<T> Cx;                                                         \
    Cx.reserve(x_nnz + y_nnz);                                                 \
    Cp.push_back(0);                                                           \
    for (int64_t i = 0; i < n_row; i++) {                                      \
      int64_t head = -2;                                                       \
      int64_t length = 0;                                                      \
                                                                               \
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
                                                                               \
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
        T result = func(A_row[head], B_row[head]);                             \
        if (result != 0) {                                                     \
          Cj.resize(nnz + 1);                                                  \
          Cj[nnz] = head;                                                      \
          Cx.resize(nnz + 1);                                                  \
          Cx[nnz] = result;                                                    \
          nnz++;                                                               \
        }                                                                      \
        int64_t temp = head;                                                   \
        head = next[head];                                                     \
        next[temp] = -1;                                                       \
        A_row[temp] = 0;                                                       \
        B_row[temp] = 0;                                                       \
      }                                                                        \
      Cp.push_back(nnz);                                                       \
    }                                                                          \
    DenseTensorMeta crows_meta(                                                \
        DataType::INT64,                                                       \
        phi::make_ddim({static_cast<int64_t>(Cp.size())}),                     \
        DataLayout::NCHW);                                                     \
    DenseTensorMeta cols_meta(                                                 \
        DataType::INT64,                                                       \
        phi::make_ddim({static_cast<int64_t>(Cj.size())}),                     \
        DataLayout::NCHW);                                                     \
    DenseTensorMeta values_meta(                                               \
        paddle::experimental::CppTypeToDataType<T>::Type(),                    \
        phi::make_ddim({static_cast<int64_t>(Cx.size())}),                     \
        DataLayout::NCHW);                                                     \
                                                                               \
    phi::DenseTensor out_crows = phi::Empty(dev_ctx, std::move(crows_meta));   \
    phi::DenseTensor out_cols = phi::Empty(dev_ctx, std::move(cols_meta));     \
    phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta)); \
    auto* out_crows_data = out_crows.mutable_data<int64_t>(place);             \
    auto* out_cols_data = out_cols.mutable_data<int64_t>(place);               \
    auto* out_values_data = out_values.mutable_data<T>(place);                 \
    std::memcpy(out_crows_data, Cp.data(), sizeof(int64_t) * Cp.size());       \
    std::memcpy(out_cols_data, Cj.data(), sizeof(int64_t) * Cj.size());        \
    std::memcpy(out_values_data, Cx.data(), sizeof(T) * Cx.size());            \
    out->SetMember(out_crows, out_cols, out_values, x.dims());                 \
  }

/*// template <typename T, typename Context, class Functor>
// void ElementWiseCsrKernel(const Context& dev_ctx,
//                           const SparseCsrTensor& x,
//                           const SparseCsrTensor& y,
//                           SparseCsrTensor* out) {
//   PADDLE_ENFORCE_EQ(x.dims(), y.dims(), "x.dim and y.dim should be same");
//   const DDim& x_dims = x.dims();
//   const auto& n_row = x_dims[0];
//   const auto& n_col = x_dims[1];
//   const auto& x_crows = x.non_zero_crows();
//   const auto& x_cols = x.non_zero_cols();
//   const auto& x_values = x.non_zero_elements();
//   const auto* x_crows_data = x_crows.data<int64_t>();
//   const auto* x_cols_data = x_cols.data<int64_t>();
//   const auto* x_values_data = x_values.data<T>();
//   const auto& y_crows = y.non_zero_crows();
//   const auto& y_cols = y.non_zero_cols();
//   const auto& y_values = y.non_zero_elements();
//   const auto* y_crows_data = y_crows.data<int64_t>();
//   const auto* y_cols_data = y_cols.data<int64_t>();
//   const auto* y_values_data = x_values.data<T>();
//   const auto place = dev_ctx.GetPlace();
//   const auto func = funcs::AddFunctor<T>();
//
//   std::vector<int64_t> next(n_col, -1);
//   std::vector<T> A_row(n_col, 0);
//   std::vector<T> B_row(n_col, 0);
//
//   int64_t nnz = 0;
//   std::vector<int64_t> Cp;
//   std::vector<int64_t> Cj;
//   std::vector<int64_t> Cx;
//   Cp[0] = 0;
//
//   for (int64_t i = 0; i < n_row; i++) {
//     int64_t head = -2;
//     int64_t length = 0;
//
//     // add a row of A to A_row
//     int64_t i_start = x_crows_data[i];
//     int64_t i_end = x_crows_data[i + 1];
//     for (int64_t jj = i_start; jj < i_end; jj++) {
//       int64_t j = x_cols_data[jj];
//
//       A_row[j] += x_values_data[jj];
//
//       if (next[j] == -1) {
//         next[j] = head;
//         head = j;
//         length++;
//       }
//     }
//
//     // add a row of B to B_row
//     i_start = y_crows_data[i];
//     i_end = y_crows_data[i + 1];
//     for (int64_t jj = i_start; jj < i_end; jj++) {
//       int64_t j = y_cols_data[jj];
//
//       B_row[j] += y_values_data[jj];
//
//       if (next[j] == -1) {
//         next[j] = head;
//         head = j;
//         length++;
//       }
//     }
//
//     // scan through columns where A or B has
//     // contributed a non-zero entry
//     for (int64_t jj = 0; jj < length; jj++) {
//       T result = func(A_row[head], B_row[head]);
//
//       if (result != 0) {
//         Cj[nnz] = head;
//         Cx[nnz] = result;
//         nnz++;
//       }
//
//       int64_t temp = head;
//       head = next[head];
//
//       next[temp] = -1;
//       A_row[temp] = 0;
//       B_row[temp] = 0;
//     }
//
//     Cp[i + 1] = nnz;
//   }
//
//   DenseTensorMeta crows_meta(
//       DataType::INT64, x.non_zero_crows().dims(), DataLayout::NCHW);
//   DenseTensorMeta cols_meta(
//       DataType::INT64, x.non_zero_cols().dims(), DataLayout::NCHW);
//   DenseTensorMeta values_meta(
//       paddle::experimental::CppTypeToDataType<T>::Type(),
//       phi::make_ddim({1, 1}),
//       DataLayout::NCHW);
//
//   phi::DenseTensor out_crows = phi::Empty(dev_ctx, std::move(crows_meta));
//   phi::DenseTensor out_cols = phi::Empty(dev_ctx, std::move(cols_meta));
//   phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));
//
//   auto* out_crows_data = out_crows.mutable_data<int64_t>(place);
//   auto* out_cols_data = out_cols.mutable_data<int64_t>(place);
//
//   std::memcpy(
//       out_crows_data, x_crows_data, sizeof(int64_t) * x_crows.dims()[0]);
//   std::memcpy(out_cols_data, x_cols_data, sizeof(int64_t) *
//   x_cols.dims()[0]); out->SetMember(out_crows, out_cols, out_values,
//   x.dims());
// }*/

/*template <typename T, typename Context>
void ElementWiseAddCsrKernel(const Context& dev_ctx,
                             const SparseCsrTensor& x,
                             const SparseCsrTensor& y,
                             SparseCsrTensor* out) {

  const DDim& x_dims = x.dims();
  const auto& n_row = x_dims[0];
  const auto& n_col = x_dims[1];
  const auto& x_crows = x.non_zero_crows();
  const auto& x_cols = x.non_zero_cols();
  const auto& x_values = x.non_zero_elements();
  const auto& x_nnz = x.non_zero_elements().numel();
  const auto* x_crows_data = x_crows.data<int64_t>();
  const auto* x_cols_data = x_cols.data<int64_t>();
  const auto* x_values_data = x_values.data<T>();
  const auto& y_crows = y.non_zero_crows();
  const auto& y_cols = y.non_zero_cols();
  const auto& y_values = y.non_zero_elements();
  const auto& y_nnz = y.non_zero_elements().numel();
  const auto* y_crows_data = y_crows.data<int64_t>();
  const auto* y_cols_data = y_cols.data<int64_t>();
  const auto* y_values_data = y_values.data<T>();
  const auto place = dev_ctx.GetPlace();
  const auto func = funcs::AddFunctor<T>();
  std::vector<int64_t> next(n_col, -1);
  std::vector<T> A_row(n_col, 0);
  std::vector<T> B_row(n_col, 0);
  int64_t nnz = 0;
  std::vector<int64_t> Cp;
  Cp.reserve(x_nnz + y_nnz);
  std::vector<int64_t> Cj;
  Cj.reserve(x_nnz + y_nnz);
  std::vector<T> Cx;
  Cx.reserve(x_nnz + y_nnz);
  //  Cp[0] = 0;
  Cp.push_back(0);
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
    for (int64_t jj = 0; jj < length; jj++) {
      T result = func(A_row[head], B_row[head]);
      if (result != 0) {
        Cj.resize(nnz + 1);
        Cj[nnz] = head;
        Cx.resize(nnz + 1);
        Cx[nnz] = result;
        nnz++;
      }
      int64_t temp = head;
      head = next[head];
      next[temp] = -1;
      A_row[temp] = 0;
      B_row[temp] = 0;
    }
    //    Cp[i + 1] = nnz;
    Cp.push_back(nnz);
  }
  DenseTensorMeta crows_meta(DataType::INT64,
                             phi::make_ddim({static_cast<int64_t>(Cp.size())}),
                             DataLayout::NCHW);
  DenseTensorMeta cols_meta(DataType::INT64,
                            phi::make_ddim({static_cast<int64_t>(Cj.size())}),
                            DataLayout::NCHW);
  DenseTensorMeta values_meta(
      paddle::experimental::CppTypeToDataType<T>::Type(),
      phi::make_ddim({static_cast<int64_t>(Cx.size())}),
      DataLayout::NCHW);

  phi::DenseTensor out_crows = phi::Empty(dev_ctx, std::move(crows_meta));
  phi::DenseTensor out_cols = phi::Empty(dev_ctx, std::move(cols_meta));
  phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));
  auto* out_crows_data = out_crows.mutable_data<int64_t>(place);
  auto* out_cols_data = out_cols.mutable_data<int64_t>(place);
  auto* out_values_data = out_values.mutable_data<T>(place);
  std::memcpy(out_crows_data, Cp.data(), sizeof(int64_t) * Cp.size());
  std::memcpy(out_cols_data, Cj.data(), sizeof(int64_t) * Cj.size());
  std::memcpy(out_values_data, Cx.data(), sizeof(T) * Cx.size());
  out->SetMember(out_crows, out_cols, out_values, x.dims());
}*/

DEFINE_CSR_ELEMENTWISE_KERNEL(Add)

DEFINE_CSR_ELEMENTWISE_KERNEL(Subtract)

DEFINE_CSR_ELEMENTWISE_KERNEL(Multiply)

template <typename T, typename Context>
void ElementWiseDivideCsrKernel(const Context& dev_ctx,
                                const SparseCsrTensor& x,
                                const SparseCsrTensor& y,
                                SparseCsrTensor* out) {
  const DDim& x_dims = x.dims();
  const auto& n_row = x_dims[0];
  const auto& n_col = x_dims[1];
  const auto& x_crows = x.non_zero_crows();
  const auto& x_cols = x.non_zero_cols();
  const auto& x_values = x.non_zero_elements();
  const auto& x_nnz = x.non_zero_elements().numel();
  const auto* x_crows_data = x_crows.data<int64_t>();
  const auto* x_cols_data = x_cols.data<int64_t>();
  const auto* x_values_data = x_values.data<T>();
  const auto& y_crows = y.non_zero_crows();
  const auto& y_cols = y.non_zero_cols();
  const auto& y_values = y.non_zero_elements();
  const auto& y_nnz = y.non_zero_elements().numel();
  const auto* y_crows_data = y_crows.data<int64_t>();
  const auto* y_cols_data = y_cols.data<int64_t>();
  const auto* y_values_data = y_values.data<T>();
  const auto place = dev_ctx.GetPlace();
  const auto func = funcs::DivideFunctor<T>();
  std::vector<int64_t> next(n_col, -1);
  std::vector<T> A_row(n_col, 0);
  std::vector<T> B_row(n_col, 0);
  int64_t nnz = 0;
  std::vector<int64_t> Cp;
  Cp.reserve(x_nnz + y_nnz);
  std::vector<int64_t> Cj;
  Cj.reserve(x_nnz + y_nnz);
  std::vector<T> Cx;
  Cx.reserve(x_nnz + y_nnz);
  Cp.push_back(0);
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
    for (int64_t jj = 0; jj < length; jj++) {
      auto result = func(A_row[head], B_row[head]);
      if (result != 0) {
        Cj.resize(nnz + 1);
        Cj[nnz] = head;
        Cx.resize(nnz + 1);
        Cx[nnz] = result;
        nnz++;
      }
      int64_t temp = head;
      head = next[head];
      next[temp] = -1;
      A_row[temp] = 0;
      B_row[temp] = 0;
    }
    Cp.push_back(nnz);
  }
  DenseTensorMeta crows_meta(DataType::INT64,
                             phi::make_ddim({static_cast<int64_t>(Cp.size())}),
                             DataLayout::NCHW);
  DenseTensorMeta cols_meta(DataType::INT64,
                            phi::make_ddim({static_cast<int64_t>(Cj.size())}),
                            DataLayout::NCHW);
  DenseTensorMeta values_meta(
      paddle::experimental::CppTypeToDataType<T>::Type(),
      phi::make_ddim({static_cast<int64_t>(Cx.size())}),
      DataLayout::NCHW);
  phi::DenseTensor out_crows = phi::Empty(dev_ctx, std::move(crows_meta));
  phi::DenseTensor out_cols = phi::Empty(dev_ctx, std::move(cols_meta));
  phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));
  auto* out_crows_data = out_crows.mutable_data<int64_t>(place);
  auto* out_cols_data = out_cols.mutable_data<int64_t>(place);
  auto* out_values_data = out_values.mutable_data<T>(place);
  std::memcpy(out_crows_data, Cp.data(), sizeof(int64_t) * Cp.size());
  std::memcpy(out_cols_data, Cj.data(), sizeof(int64_t) * Cj.size());
  std::memcpy(out_values_data, Cx.data(), sizeof(T) * Cx.size());
  out->SetMember(out_crows, out_cols, out_values, x.dims());
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(CSR_ELEMENTWISE_API_NAME(add),
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CSR_ELEMENTWISE_KERNEL_NAME(Add),
                   float,
                   double) {}

PD_REGISTER_KERNEL(CSR_ELEMENTWISE_API_NAME(sub),
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CSR_ELEMENTWISE_KERNEL_NAME(Subtract),
                   float,
                   double) {}

PD_REGISTER_KERNEL(CSR_ELEMENTWISE_API_NAME(mul),
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CSR_ELEMENTWISE_KERNEL_NAME(Multiply),
                   float,
                   double) {}

PD_REGISTER_KERNEL(CSR_ELEMENTWISE_API_NAME(div),
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CSR_ELEMENTWISE_KERNEL_NAME(Divide),
                   float,
                   double) {}