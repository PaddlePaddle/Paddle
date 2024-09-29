/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"

#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/funcs/sparse/common_shape.h"

namespace phi::sparse {

template <typename T>
inline bool IsZero(const T* data, const size_t n) {
  const T zero = static_cast<T>(0);
  for (size_t i = 0; i < n; i++) {
    if (data[i] != zero) {
      return false;
    }
  }
  return true;
}

// TODO(zhangkaihuo): implement a kernel to count the number of non-zero
// elements in tensor
template <typename T>
inline int64_t GetNonZeroNum(const DenseTensor& dense,
                             const int64_t sparse_dim) {
  const auto& dims = dense.dims();
  PADDLE_ENFORCE_GE(
      dims.size(),
      sparse_dim,
      common::errors::InvalidArgument(
          "sparse_dim(%d) should be less than or equal to dense.dim(%d)",
          sparse_dim,
          dims.size()));

  auto dims_2d = flatten_to_2d(dims, static_cast<int>(sparse_dim));
  const int rows = static_cast<int>(dims_2d[0]);
  const int cols = static_cast<int>(dims_2d[1]);

  const T* data = dense.data<T>();
  int64_t non_zero_num = 0;
  for (int64_t i = 0; i < rows; i++) {
    if (!IsZero(data + i * cols, cols)) {
      non_zero_num = non_zero_num + 1;
    }
  }
  return non_zero_num;
}

template <typename T, typename Context>
void DenseToCooKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const int64_t sparse_dim,
                      SparseCooTensor* out) {
  const T* x_data = x.data<T>();
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_LE(sparse_dim,
                    x_dims.size(),
                    common::errors::InvalidArgument(
                        "sparse_dim must be less than the size of x.dims()"));
  PADDLE_ENFORCE_GT(
      sparse_dim, 0, common::errors::InvalidArgument("sparse_dim must be >0"));

  int64_t non_zero_num = GetNonZeroNum<T>(x, sparse_dim);

  const auto values_dims =
      phi::funcs::sparse::InferDenseDims(x_dims, sparse_dim, non_zero_num);
  DenseTensorMeta values_meta(x.meta().dtype, values_dims, x.meta().layout);
  phi::DenseTensor indices =
      phi::Empty<int64_t>(dev_ctx, {sparse_dim, non_zero_num});
  phi::DenseTensor values = phi::Empty(dev_ctx, std::move(values_meta));
  int64_t* indices_data = indices.data<int64_t>();
  T* values_data = values.data<T>();

  auto dims_2d = flatten_to_2d(x_dims, static_cast<int>(sparse_dim));
  const int rows = static_cast<int>(dims_2d[0]);
  const int cols = static_cast<int>(dims_2d[1]);

  int index = 0;
  for (int i = 0; i < rows; i++) {
    if (!IsZero(x_data + i * cols, cols)) {
      int64_t sparse_index = i;
      for (int j = static_cast<int>(sparse_dim - 1); j >= 0; j--) {
        indices_data[j * non_zero_num + index] = sparse_index % x_dims[j];
        sparse_index /= x_dims[j];
      }
      memcpy(values_data + index * cols, x_data + i * cols, cols * sizeof(T));
      ++index;
    }
  }

  out->SetMember(indices, values, x_dims, true);
}

template <typename T, typename IntT>
void CsrToCooCPUKernel(const CPUContext& dev_ctx,
                       const SparseCsrTensor& x,
                       SparseCooTensor* out) {
  const DDim& x_dims = x.dims();
  const int64_t non_zero_num = x.cols().numel();
  int64_t sparse_dim = 2;
  if (x_dims.size() == 3) {
    sparse_dim = 3;
  }
  phi::DenseTensor indices =
      phi::Empty<IntT>(dev_ctx, {sparse_dim, non_zero_num});
  phi::DenseTensor values = phi::Empty<T>(dev_ctx, {non_zero_num});
  if (x.nnz() <= 0) {
    out->SetMember(indices, values, x_dims, true);
    return;
  }
  const auto& csr_crows = x.crows();
  const auto& csr_cols = x.cols();
  const auto& csr_values = x.values();
  const IntT* csr_crows_data = csr_crows.data<IntT>();
  const IntT* csr_cols_data = csr_cols.data<IntT>();
  const T* csr_values_data = csr_values.data<T>();

  IntT* coo_indices = indices.data<IntT>();
  IntT* batch_ptr = x_dims.size() == 2 ? nullptr : coo_indices;
  IntT* coo_rows_data =
      x_dims.size() == 2 ? coo_indices : batch_ptr + non_zero_num;
  IntT* coo_cols_data = coo_rows_data + non_zero_num;
  T* coo_values_data = values.data<T>();

  int batch = static_cast<int>(x_dims.size() == 2 ? 1 : x_dims[0]);
  int rows = static_cast<int>(x_dims.size() == 2 ? x_dims[0] : x_dims[1]);

  int index = 0;
  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < rows; i++) {
      for (IntT j = csr_crows_data[b * (rows + 1) + i];
           j < csr_crows_data[b * (rows + 1) + i + 1];
           j++) {
        coo_rows_data[index] = i;
        if (batch_ptr) {
          batch_ptr[index] = b;
        }
        ++index;
      }
    }
  }

  memcpy(coo_cols_data, csr_cols_data, sizeof(IntT) * non_zero_num);
  memcpy(coo_values_data, csr_values_data, sizeof(T) * non_zero_num);
  out->SetMember(indices, values, x_dims, true);
}

template <typename T, typename Context>
void CsrToCooKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    SparseCooTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.crows().dtype(), "CsrToCooCPUKernel", ([&] {
                                 CsrToCooCPUKernel<T, data_t>(dev_ctx, x, out);
                               }));
}

template <typename T, typename IntT>
void CooToCsrCPUKernel(const CPUContext& dev_ctx,
                       const SparseCooTensor& x,
                       SparseCsrTensor* out) {
  const auto& x_dims = x.dims();
  bool valid = x_dims.size() == 2 || x_dims.size() == 3;
  PADDLE_ENFORCE_EQ(valid,
                    true,
                    common::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D or 3-D matrix"));
  const int64_t non_zero_num = x.nnz();

  int batches = static_cast<int>(x_dims.size() == 2 ? 1 : x_dims[0]);
  int rows = static_cast<int>(x_dims.size() == 2 ? x_dims[0] : x_dims[1]);

  phi::DenseTensor crows = phi::Empty<IntT>(dev_ctx, {batches * (rows + 1)});
  phi::DenseTensor cols = phi::Empty<IntT>(dev_ctx, {non_zero_num});
  phi::DenseTensor values = phi::EmptyLike<T, CPUContext>(dev_ctx, x.values());
  if (non_zero_num <= 0) {
    out->SetMember(crows, cols, values, x_dims);
    return;
  }
  IntT* csr_crows_data = crows.data<IntT>();
  IntT* csr_cols_data = cols.data<IntT>();
  T* csr_values_data = values.data<T>();

  const auto& coo_indices = x.indices();
  const auto& coo_values = x.values();
  const IntT* batches_ptr = coo_indices.data<IntT>();
  const IntT* coo_rows_data =
      x_dims.size() == 2 ? batches_ptr : batches_ptr + non_zero_num;
  const IntT* coo_cols_data = coo_rows_data + non_zero_num;
  const T* coo_values_data = coo_values.data<T>();

  std::vector<int64_t> offsets(batches, 0);
  if (batches > 1) {
    for (int i = 0; i < non_zero_num; i++) {
      if (i == non_zero_num - 1 || batches_ptr[i] != batches_ptr[i + 1]) {
        const int start = batches_ptr[i];
        const int end = i == non_zero_num - 1 ? batches : batches_ptr[i + 1];
        for (int j = start; j < end; j++) {
          offsets[j] = i + 1;
        }
      }
    }
  } else {
    offsets[0] = non_zero_num;
  }

  for (int b = 0; b < batches; b++) {
    int batch_start = 0;
    int batch_non_zero_num = static_cast<int>(offsets[b]);
    if (b > 0) {
      batch_start = static_cast<int>(offsets[b - 1]);
      batch_non_zero_num -= batch_start;
    }
    auto* coo_rows_ptr = coo_rows_data + batch_start;
    for (int i = 0; i <= coo_rows_ptr[0]; i++) {
      csr_crows_data[b * (rows + 1) + i] = 0;
    }
    for (int64_t i = 1; i < batch_non_zero_num; i++) {
      for (IntT j = coo_rows_ptr[i - 1]; j < coo_rows_ptr[i]; j++) {
        csr_crows_data[b * (rows + 1) + j + 1] = i;
      }
    }
    for (IntT i = coo_rows_ptr[batch_non_zero_num - 1] + 1; i < rows + 1; i++) {
      csr_crows_data[b * (rows + 1) + i] = batch_non_zero_num;
    }
    if (batch_non_zero_num == 0) {
      memset(csr_crows_data + b * (rows + 1), 0, sizeof(IntT) * (rows + 1));
    }
  }

  memcpy(csr_cols_data, coo_cols_data, sizeof(IntT) * non_zero_num);
  memcpy(csr_values_data, coo_values_data, sizeof(T) * non_zero_num);
  out->SetMember(crows, cols, values, x_dims);
}

template <typename T, typename Context>
void CooToCsrKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    SparseCsrTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "CooToCsrCPUKernel", ([&] {
                                 CooToCsrCPUKernel<T, data_t>(dev_ctx, x, out);
                               }));
}

template <typename T, typename IntT>
void CooToDenseCPUKernel(const CPUContext& dev_ctx,
                         const SparseCooTensor& x,
                         DenseTensor* out) {
  const auto non_zero_num = x.nnz();
  const auto& dense_dims = x.dims();
  const auto& indices = x.indices();
  const auto& values = x.values();
  const auto indices_dims = common::vectorize<int>(indices.dims());
  int64_t sparse_dim = indices_dims[0];
  if (indices_dims.size() == 1) {
    sparse_dim = 1;
  }
  const int64_t dense_dim = x.dense_dim();

  const T* x_data = values.data<T>();
  dev_ctx.template Alloc<T>(out);
  T* out_data = out->data<T>();
  memset(out_data, 0, sizeof(T) * out->numel());

  if (x.nnz() <= 0) {
    return;
  }

  int64_t base_offset = 1;
  for (int64_t i = 0; i < dense_dim; i++) {
    base_offset *= dense_dims[static_cast<int>(sparse_dim + i)];
  }
  std::vector<int64_t> sparse_offsets(sparse_dim);
  int64_t offset = 1;
  for (int i = static_cast<int>(sparse_dim - 1); i >= 0; i--) {
    sparse_offsets[i] = offset;
    offset *= dense_dims[i];
  }

  for (auto i = 0; i < non_zero_num; i++) {
    int64_t index = 0;
    for (int j = 0; j < sparse_dim; j++) {
      index += indices.data<IntT>()[j * non_zero_num + i] * sparse_offsets[j];
    }

    for (int j = 0; j < base_offset; j++) {
      out_data[index * base_offset + j] = x_data[i * base_offset + j];
    }
  }
}

template <typename T, typename Context>
void CooToDenseKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      DenseTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "CooToDenseCPUKernel", ([&] {
        CooToDenseCPUKernel<T, data_t>(dev_ctx, x, out);
      }));
}

}  // namespace phi::sparse

PD_REGISTER_KERNEL(dense_to_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::DenseToCooKernel,
                   float,
                   double,
                   paddle::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(csr_to_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrToCooKernel,
                   float,
                   double,
                   paddle::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(coo_to_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CooToCsrKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(dense_to_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::DenseToCsrKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(coo_to_dense,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CooToDenseKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(csr_to_dense,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrToDenseKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_REGISTER_KERNEL(values_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ValuesCooKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(indices_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::IndicesCooKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(values_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::ValuesCsrKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

PD_REGISTER_KERNEL(sparse_coo_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooTensorKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
