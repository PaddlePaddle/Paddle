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
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/sparse/cpu/sparse_utils_kernel.cc"
#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename IntT, typename Context, typename Functor>
void ElementWiseKernelImpl(const Context& dev_ctx,
                           const SparseCsrTensor& x,
                           const SparseCsrTensor& y,
                           SparseCsrTensor* out,
                           const Functor& functor) {
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
  const auto* x_crows_data = x.non_zero_crows().data<IntT>();
  const auto* x_cols_data = x.non_zero_cols().data<IntT>();
  const auto* x_values_data = x.non_zero_elements().data<T>();
  const auto& y_nnz = y.non_zero_elements().numel();
  const auto* y_crows_data = y.non_zero_crows().data<IntT>();
  const auto* y_cols_data = y.non_zero_cols().data<IntT>();
  const auto* y_values_data = y.non_zero_elements().data<T>();

  std::vector<IntT> next(n_col, -1);
  std::vector<T> A_row(n_col, 0);
  std::vector<T> B_row(n_col, 0);

  std::vector<IntT> out_crows_vec;
  std::vector<IntT> out_cols_vec;
  std::vector<T> out_values_vec;

  std::vector<IntT> out_batch_crows_vec;
  std::vector<IntT> out_batch_cols_vec;
  std::vector<T> out_batch_values_vec;
  out_batch_crows_vec.reserve(x_nnz + y_nnz);
  out_batch_cols_vec.reserve(x_nnz + y_nnz);
  out_batch_values_vec.reserve(x_nnz + y_nnz);

  IntT x_prev_batch_nnz = 0;
  IntT y_prev_batch_nnz = 0;

  IntT nnz = 0;
  //  merge two batches
  for (IntT b = 0; b < n_batch; b++) {
    out_batch_crows_vec.push_back(0);
    IntT x_batch_nnz = 0;
    IntT y_batch_nnz = 0;
    for (IntT i = 0; i < n_row; i++) {
      IntT x_idx = x_crows_data[i + b * (n_row + 1)];
      IntT y_idx = y_crows_data[i + b * (n_row + 1)];

      IntT x_end = x_crows_data[i + b * (n_row + 1) + 1];
      IntT y_end = y_crows_data[i + b * (n_row + 1) + 1];
      x_batch_nnz += (x_end - x_idx);
      y_batch_nnz += (y_end - y_idx);

      while (x_idx < x_end && y_idx < y_end) {
        IntT A_j = x_cols_data[x_idx + x_prev_batch_nnz];
        IntT B_j = y_cols_data[y_idx + y_prev_batch_nnz];

        if (A_j == B_j) {
          T result = functor(x_values_data[x_idx + x_prev_batch_nnz],
                             y_values_data[y_idx + y_prev_batch_nnz]);
          if (result != 0) {
            out_batch_cols_vec.push_back(A_j);
            out_batch_values_vec.push_back(result);
            nnz++;
          }
          x_idx++;
          y_idx++;
        } else if (A_j < B_j) {
          T result = functor(x_values_data[x_idx + x_prev_batch_nnz], 0);
          if (result != 0) {
            out_batch_cols_vec.push_back(A_j);
            out_batch_values_vec.push_back(result);
            nnz++;
          }
          x_idx++;
        } else {
          T result = functor(0, y_values_data[y_idx + y_prev_batch_nnz]);
          if (result != 0) {
            out_batch_cols_vec.push_back(B_j);
            out_batch_values_vec.push_back(result);
            nnz++;
          }
          y_idx++;
        }
      }

      while (x_idx < x_end) {
        T result = functor(x_values_data[x_idx + x_prev_batch_nnz], 0);
        if (result != 0) {
          out_batch_cols_vec.push_back(x_cols_data[x_idx + x_prev_batch_nnz]);
          out_batch_values_vec.push_back(result);
          nnz++;
        }
        x_idx++;
      }
      while (y_idx < y_end) {
        T result = functor(0, y_values_data[y_idx + y_prev_batch_nnz]);
        if (result != 0) {
          out_batch_cols_vec.push_back(y_cols_data[y_idx + y_prev_batch_nnz]);
          out_batch_values_vec.push_back(result);
          nnz++;
        }
        y_idx++;
      }
      out_batch_crows_vec.push_back(nnz);
    }
    nnz = 0;
    x_prev_batch_nnz += x_batch_nnz;
    y_prev_batch_nnz += y_batch_nnz;
    out_cols_vec.insert(out_cols_vec.end(),
                        out_batch_cols_vec.begin(),
                        out_batch_cols_vec.end());
    out_crows_vec.insert(out_crows_vec.end(),
                         out_batch_crows_vec.begin(),
                         out_batch_crows_vec.end());
    out_values_vec.insert(out_values_vec.end(),
                          out_batch_values_vec.begin(),
                          out_batch_values_vec.end());
    out_batch_cols_vec.clear();
    out_batch_crows_vec.clear();
    out_batch_values_vec.clear();
  }

  //  out_crows_vec.resize(out_crows_vec.size() - 1);
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
  std::memcpy(out_crows.template data<IntT>(),
              out_crows_vec.data(),
              sizeof(IntT) * out_crows_vec.size());
  std::memcpy(out_cols.template data<IntT>(),
              out_cols_vec.data(),
              sizeof(IntT) * out_cols_vec.size());
  std::memcpy(out_values.template data<T>(),
              out_values_vec.data(),
              sizeof(T) * out_values_vec.size());
  out->SetMember(out_crows, out_cols, out_values, x.dims());
}

// coordinate a equal to coordinate b
template <typename IntT>
struct ColEqual {
  ColEqual(const IntT& row, const IntT& col_A, const IntT& col_B)
      : row(row), col_A(col_A), col_B(col_B) {}
  inline bool operator()(const IntT* a,
                         const IntT* b,
                         IntT idx_a,
                         IntT idx_b) {
    for (int i = 0; i < row; ++i) {
      if (a[idx_a + i * col_A] != b[idx_b + i * col_B]) {
        return false;
      }
    }
    return true;
  }
  const IntT& row;
  const IntT& col_A;
  const IntT& col_B;
};

// coordinate a less than coordinate b
template <typename IntT>
struct ColLess {
  ColLess(const IntT& row, const IntT& col_A, const IntT& col_B)
      : row(row), col_A(col_A), col_B(col_B) {}
  inline bool operator()(const IntT* a,
                         const IntT* b,
                         IntT idx_a,
                         IntT idx_b) {
    for (int i = 0; i < row; ++i) {
      if (a[idx_a + i * col_A] == b[idx_b + i * col_B]) {
        continue;
      } else {
        return a[idx_a + i * col_A] < b[idx_b + i * col_B];
      }
    }
    return false;
  }
  const IntT& row;
  const IntT& col_A;
  const IntT& col_B;
};

template <class InputIterator>
bool IsZeroElement(InputIterator first, InputIterator last) {
  for (; first != last; ++first) {
    if (*first != 0) {
      return false;
    }
  }
  return true;
}

// SparseCooTensor elwise op, only support dims >= 3
template <typename T, typename IntT, typename Context, typename Functor>
void ElementWiseCooKernelImpl(const Context& dev_ctx,
                              const SparseCooTensor& x,
                              const SparseCooTensor& y,
                              SparseCooTensor* out,
                              const Functor& functor) {
  PADDLE_ENFORCE_EQ(x.dims(),
                    y.dims(),
                    phi::errors::InvalidArgument(
                        "The input tensor X's shape "
                        "should be identical with Y's shape. But received X's "
                        "shape = [%s], Y's shape = [%s].",
                        x.dims(),
                        y.dims()));
  PADDLE_ENFORCE_GE(x.dims().size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The input tensor dims should be at least 3D. But "
                        "received X's "
                        "dim = [%s], Y's dim = [%s].",
                        x.dims().size(),
                        y.dims().size()));
  int64_t element_size = 1;
  for (int j = 1; j < x.non_zero_elements().dims().size(); ++j) {
    element_size *= x.non_zero_elements().dims()[j];
  }
  const auto nnz_x = x.non_zero_elements().dims()[0];
  const auto nnz_y = y.non_zero_elements().dims()[0];
  const auto x_indices = x.non_zero_indices().data<IntT>();
  const auto y_indices = y.non_zero_indices().data<IntT>();
  const auto x_values = x.non_zero_elements().data<T>();
  const auto y_values = y.non_zero_elements().data<T>();
  const auto indices_dim = x.non_zero_indices().dims()[0];
  ColEqual<IntT> ceq = ColEqual<IntT>(indices_dim, nnz_x, nnz_y);
  ColLess<IntT> cle = ColLess<IntT>(indices_dim, nnz_x, nnz_y);

  std::vector<std::vector<IntT>> out_indices_vec;
  std::vector<T> out_values_vec;
  out_indices_vec.reserve(std::max(nnz_x, nnz_y));
  out_values_vec.reserve(std::max(nnz_x, nnz_y));

  for (int j = 0; j < indices_dim; ++j) {
    out_indices_vec.push_back(std::vector<IntT>());
  }

  IntT nnz = 0;
  IntT a = 0;
  IntT b = 0;
  //  merge x and y
  for (int i = 0; i < std::max(nnz_x, nnz_y); ++i) {
    // coordinate x[a] = coordinate y[b]
    if (ceq(x_indices, y_indices, a, b) && a < nnz_x && b < nnz_y) {
      std::vector<T> result;
      result.reserve(element_size);
      for (int j = 0; j < element_size; ++j) {
        result.push_back(functor(x_values[a * element_size + j],
                                 y_values[b * element_size + j]));
      }
      if (!sparse::IsZeroElement(result.begin(), result.end())) {
        for (auto j = 0; j < indices_dim; ++j) {
          out_indices_vec[j].push_back(x_indices[j * nnz_x + a]);
        }
        std::for_each(result.begin(), result.end(), [&out_values_vec](T& x) {
          out_values_vec.push_back(x);
        });
        ++nnz;
      }
      ++a;
      ++b;
    }
    // coordinate x[a] < coordinate y[b]
    else if ((a < nnz_x && b >= nnz_y) || cle(x_indices, y_indices, a, b)) {
      std::vector<T> result;
      result.reserve(element_size);
      for (int j = 0; j < element_size; ++j) {
        result.push_back(functor(x_values[a * element_size + j], 0));
      }
      if (!sparse::IsZeroElement(result.begin(), result.end())) {
        for (auto j = 0; j < indices_dim; ++j) {
          out_indices_vec[j].push_back(x_indices[j * nnz_x + a]);
        }
        std::for_each(result.begin(), result.end(), [&out_values_vec](T& x) {
          out_values_vec.push_back(x);
        });
        ++nnz;
      }
      ++a;
    }
    // coordinate x[a] > coordinate y[b]
    else if ((a >= nnz_x && b < nnz_y) || cle(y_indices, x_indices, b, a)) {
      std::vector<T> result;
      result.reserve(element_size);
      for (int j = 0; j < element_size; ++j) {
        result.push_back(functor(0, y_values[b * element_size + j]));
      }
      if (!sparse::IsZeroElement(result.begin(), result.end())) {
        for (auto j = 0; j < indices_dim; ++j) {
          out_indices_vec[j].push_back(y_indices[j * nnz_x + a]);
        }
        std::for_each(result.begin(), result.end(), [&out_values_vec](T& x) {
          out_values_vec.push_back(x);
        });
        ++nnz;
      }
      ++b;
    }
  }

  if (nnz == 0) {
    phi::DenseTensor out_indices =
        phi::EmptyLike<IntT>(dev_ctx, x.non_zero_indices());
    phi::DenseTensor out_values =
        phi::EmptyLike<T>(dev_ctx, x.non_zero_elements());
    out->SetMember(out_indices, out_values, x.dims());
  } else {
    DenseTensorMeta indices_meta(
        paddle::experimental::CppTypeToDataType<IntT>::Type(),
        phi::make_ddim({static_cast<int64_t>(indices_dim),
                        static_cast<int64_t>(out_indices_vec[0].size())}),
        DataLayout::NCHW);
    auto indeces_dim = vectorize(slice_ddim(
        x.non_zero_elements().dims(), 1, x.non_zero_elements().dims().size()));
    indeces_dim.insert(indeces_dim.begin(), nnz);
    DenseTensorMeta values_meta(
        paddle::experimental::CppTypeToDataType<T>::Type(),
        phi::make_ddim(indeces_dim),
        DataLayout::NCHW);
    phi::DenseTensor out_indices = phi::Empty(dev_ctx, std::move(indices_meta));
    phi::DenseTensor out_values = phi::Empty(dev_ctx, std::move(values_meta));

    for (auto j = 0; j < out_indices_vec.size(); ++j) {
      auto* indices_ptr =
          out_indices.template data<IntT>() + j * out_indices_vec[j].size();
      std::copy(
          out_indices_vec[j].begin(), out_indices_vec[j].end(), indices_ptr);
    }

    std::memcpy(out_values.template data<T>(),
                out_values_vec.data(),
                sizeof(T) * out_values_vec.size());
    out->SetMember(out_indices, out_values, x.dims());
  }
}

#define DEFINE_CSR_ELEMENTWISE_CPU_KERNEL(name)                       \
  template <typename T, typename IntT, typename Context>              \
  void ElementWise##name##CsrCPUKernel(const Context& dev_ctx,        \
                                       const SparseCsrTensor& x,      \
                                       const SparseCsrTensor& y,      \
                                       SparseCsrTensor* out) {        \
    funcs::name##Functor<T> functor;                                  \
    ElementWiseKernelImpl<T, IntT, Context, funcs::name##Functor<T>>( \
        dev_ctx, x, y, out, functor);                                 \
  }

#define DEFINE_CSR_ELEMENTWISE_KERNEL(name)                                   \
  template <typename T, typename Context>                                     \
  void ElementWise##name##CsrKernel(const Context& dev_ctx,                   \
                                    const SparseCsrTensor& x,                 \
                                    const SparseCsrTensor& y,                 \
                                    SparseCsrTensor* out) {                   \
    PD_VISIT_INTEGRAL_TYPES(                                                  \
        x.non_zero_crows().dtype(), "ElementWise##name##CsrCPUKernel", ([&] { \
          ElementWise##name##CsrCPUKernel<T, data_t>(dev_ctx, x, y, out);     \
        }));                                                                  \
  }

#define DEFINE_ELEMENTWISE_COO_CPU_KERNEL(name)                          \
  template <typename T, typename IntT, typename Context>                 \
  void ElementWise##name##CooCPUKernel(const Context& dev_ctx,           \
                                       const SparseCooTensor& x,         \
                                       const SparseCooTensor& y,         \
                                       SparseCooTensor* out) {           \
    funcs::name##Functor<T> functor;                                     \
    ElementWiseCooKernelImpl<T, IntT, Context, funcs::name##Functor<T>>( \
        dev_ctx, x, y, out, functor);                                    \
  }

#define DEFINE_ELEMENTWISE_COO_KERNEL(name)                                 \
  template <typename T, typename Context>                                   \
  void ElementWise##name##CooKernel(const Context& dev_ctx,                 \
                                    const SparseCooTensor& x,               \
                                    const SparseCooTensor& y,               \
                                    SparseCooTensor* out) {                 \
    if (x.dims().size() < 3) {                                              \
      const auto csr_x = SparseCooToCsr<T>(dev_ctx, x);                     \
      const auto csr_y = SparseCooToCsr<T>(dev_ctx, y);                     \
      DenseTensor non_zero_crows;                                           \
      DenseTensor non_zero_cols;                                            \
      DenseTensor non_zero_elements;                                        \
      SparseCsrTensor csr_out(                                              \
          non_zero_crows, non_zero_cols, non_zero_elements, x.dims());      \
      PD_VISIT_INTEGRAL_TYPES(csr_x.non_zero_crows().dtype(),               \
                              "ElementWise##name##CsrCPUKernel",            \
                              ([&] {                                        \
                                ElementWise##name##CsrCPUKernel<T, data_t>( \
                                    dev_ctx, csr_x, csr_y, &csr_out);       \
                              }));                                          \
      *out = SparseCsrToCoo<T>(dev_ctx, csr_out);                           \
    } else {                                                                \
      PD_VISIT_INTEGRAL_TYPES(x.non_zero_indices().dtype(),                 \
                              "ElementWise##name##CooCPUKernel",            \
                              ([&] {                                        \
                                ElementWise##name##CooCPUKernel<T, data_t>( \
                                    dev_ctx, x, y, out);                    \
                              }));                                          \
    }                                                                       \
  }

DEFINE_CSR_ELEMENTWISE_CPU_KERNEL(Add)
DEFINE_CSR_ELEMENTWISE_CPU_KERNEL(Subtract)
DEFINE_CSR_ELEMENTWISE_CPU_KERNEL(Multiply)
DEFINE_CSR_ELEMENTWISE_CPU_KERNEL(Divide)

DEFINE_CSR_ELEMENTWISE_KERNEL(Add)
DEFINE_CSR_ELEMENTWISE_KERNEL(Subtract)
DEFINE_CSR_ELEMENTWISE_KERNEL(Multiply)
DEFINE_CSR_ELEMENTWISE_KERNEL(Divide)

DEFINE_ELEMENTWISE_COO_CPU_KERNEL(Add)
DEFINE_ELEMENTWISE_COO_CPU_KERNEL(Subtract)
DEFINE_ELEMENTWISE_COO_CPU_KERNEL(Multiply)
DEFINE_ELEMENTWISE_COO_CPU_KERNEL(Divide)

DEFINE_ELEMENTWISE_COO_KERNEL(Add)
DEFINE_ELEMENTWISE_COO_KERNEL(Subtract)
DEFINE_ELEMENTWISE_COO_KERNEL(Multiply)
DEFINE_ELEMENTWISE_COO_KERNEL(Divide)

}  // namespace sparse
}  // namespace phi

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
