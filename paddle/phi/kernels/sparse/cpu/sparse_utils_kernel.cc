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

#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/sparse/common_shape.h"

namespace phi {
namespace sparse {

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
      phi::errors::InvalidArgument(
          "sparse_dim(%d) should be less than or equal to dense.dim(%d)",
          sparse_dim,
          dims.size()));

  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];

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
void DenseToSparseCooKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const int64_t sparse_dim,
                            SparseCooTensor* out) {
  const T* x_data = x.data<T>();
  const auto& x_dims = x.dims();

  int64_t non_zero_num = GetNonZeroNum<T>(x, sparse_dim);

  const auto place = dev_ctx.GetPlace();
  const auto values_dims =
      phi::funcs::sparse::InferDenseDims(x_dims, sparse_dim, non_zero_num);
  DenseTensorMeta indices_meta(DataType::INT64,
                               {sparse_dim, static_cast<int64_t>(non_zero_num)},
                               DataLayout::NCHW);
  DenseTensorMeta values_meta(x.meta().dtype, values_dims, x.meta().layout);
  phi::DenseTensor indices = phi::Empty(dev_ctx, std::move(indices_meta));
  phi::DenseTensor values = phi::Empty(dev_ctx, std::move(values_meta));
  int64_t* indices_data = indices.mutable_data<int64_t>(place);
  T* values_data = values.mutable_data<T>(place);

  auto dims_2d = flatten_to_2d(x_dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];

  int index = 0;
  for (int i = 0; i < rows; i++) {
    if (!IsZero(x_data + i * cols, cols)) {
      int64_t sparse_index = i;
      for (int64_t j = sparse_dim - 1; j >= 0; j--) {
        indices_data[j * non_zero_num + index] = sparse_index % x_dims[j];
        sparse_index /= x_dims[j];
      }
      memcpy(values_data + index * cols, x_data + i * cols, cols * sizeof(T));
      ++index;
    }
  }
  out->SetMember(indices, values, x_dims, true);
}

template <typename T, typename Context>
void SparseCsrToCooKernel(const Context& dev_ctx,
                          const SparseCsrTensor& x,
                          SparseCooTensor* out) {
  const DDim& x_dims = x.dims();
  const int64_t non_zero_num = x.non_zero_cols().numel();
  const auto& csr_crows = x.non_zero_crows();
  const auto& csr_cols = x.non_zero_cols();
  const auto& csr_values = x.non_zero_elements();
  const int64_t* csr_crows_data = csr_crows.data<int64_t>();
  const int64_t* csr_cols_data = csr_cols.data<int64_t>();
  const T* csr_values_data = csr_values.data<T>();

  int64_t sparse_dim = 2;
  if (x_dims.size() == 3) {
    sparse_dim = 3;
  }
  const auto place = dev_ctx.GetPlace();
  DenseTensorMeta indices_meta(
      DataType::INT64, {sparse_dim, non_zero_num}, DataLayout::NCHW);
  DenseTensorMeta values_meta(
      x.dtype(), {non_zero_num}, x.non_zero_elements().layout());
  phi::DenseTensor indices = phi::Empty(dev_ctx, std::move(indices_meta));
  phi::DenseTensor values = phi::Empty(dev_ctx, std::move(values_meta));
  int64_t* coo_indices = indices.mutable_data<int64_t>(place);
  int64_t* batch_ptr = x_dims.size() == 2 ? nullptr : coo_indices;
  int64_t* coo_rows_data =
      x_dims.size() == 2 ? coo_indices : batch_ptr + non_zero_num;
  int64_t* coo_cols_data = coo_rows_data + non_zero_num;
  T* coo_values_data = values.mutable_data<T>(place);

  int batch = x_dims.size() == 2 ? 1 : x_dims[0];
  int rows = x_dims.size() == 2 ? x_dims[0] : x_dims[1];

  int index = 0;
  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < rows; i++) {
      for (int j = csr_crows_data[b * (rows + 1) + i];
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

  memcpy(coo_cols_data, csr_cols_data, sizeof(int64_t) * non_zero_num);
  memcpy(coo_values_data, csr_values_data, sizeof(T) * non_zero_num);
  out->SetMember(indices, values, x_dims, true);
}

template <typename T, typename Context>
void SparseCooToCsrKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          SparseCsrTensor* out) {
  const auto& x_dims = x.dims();
  bool valid = x_dims.size() == 2 || x_dims.size() == 3;
  PADDLE_ENFORCE_EQ(valid,
                    true,
                    phi::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D or 3-D matrix"));
  const int64_t non_zero_num = x.nnz();
  if (non_zero_num <= 0) return;

  int batchs = x_dims.size() == 2 ? 1 : x_dims[0];
  int rows = x_dims.size() == 2 ? x_dims[0] : x_dims[1];

  const auto place = dev_ctx.GetPlace();
  DenseTensorMeta crows_meta(
      DataType::INT64, {batchs * (rows + 1)}, DataLayout::NCHW);
  DenseTensorMeta cols_meta(DataType::INT64, {non_zero_num}, DataLayout::NCHW);
  DenseTensorMeta values_meta(
      x.dtype(), {non_zero_num}, x.non_zero_elements().layout());
  phi::DenseTensor non_zero_crows(
      phi::make_intrusive<paddle::experimental::SharedStorage>(place),
      std::move(crows_meta));
  phi::DenseTensor non_zero_cols(
      phi::make_intrusive<paddle::experimental::SharedStorage>(place),
      std::move(cols_meta));
  phi::DenseTensor non_zero_elements(
      phi::make_intrusive<paddle::experimental::SharedStorage>(place),
      std::move(values_meta));
  int64_t* csr_crows_data = non_zero_crows.mutable_data<int64_t>(place);
  int64_t* csr_cols_data = non_zero_cols.mutable_data<int64_t>(place);
  T* csr_values_data = non_zero_elements.mutable_data<T>(place);

  const auto& coo_indices = x.non_zero_indices();
  const auto& coo_values = x.non_zero_elements();
  const int64_t* batchs_ptr = coo_indices.data<int64_t>();
  const int64_t* coo_rows_data =
      batchs == 1 ? batchs_ptr : batchs_ptr + non_zero_num;
  const int64_t* coo_cols_data = coo_rows_data + non_zero_num;
  const T* coo_values_data = coo_values.data<T>();

  if (!x.coalesced()) {
    // TODO(zhangkahuo): call coalesced() to distinct and sort the indices
  }

  std::vector<int64_t> offsets(batchs, 0);
  if (batchs > 1) {
    for (int i = 0; i < non_zero_num; i++) {
      if (i == non_zero_num - 1 || batchs_ptr[i] != batchs_ptr[i + 1]) {
        offsets[batchs_ptr[i]] = i + 1;
      }
    }
  } else {
    offsets[0] = non_zero_num;
  }

  for (int b = 0; b < batchs; b++) {
    if (offsets[b] == 0) continue;
    int batch_start = 0;
    int batch_non_zero_num = offsets[b];
    if (b > 0) {
      batch_start = offsets[b - 1];
      batch_non_zero_num -= batch_start;
    }
    auto* coo_rows_ptr = coo_rows_data + batch_start;
    for (int i = 0; i <= coo_rows_ptr[0]; i++) {
      csr_crows_data[b * (rows + 1) + i] = 0;
    }
    for (int64_t i = 1; i < batch_non_zero_num; i++) {
      for (int j = coo_rows_ptr[i - 1]; j < coo_rows_ptr[i]; j++) {
        csr_crows_data[b * (rows + 1) + j + 1] = i;
      }
    }
    for (int64_t i = coo_rows_ptr[batch_non_zero_num - 1] + 1; i < rows + 1;
         i++) {
      csr_crows_data[b * (rows + 1) + i] = batch_non_zero_num;
    }
  }

  memcpy(csr_cols_data, coo_cols_data, sizeof(int64_t) * non_zero_num);
  memcpy(csr_values_data, coo_values_data, sizeof(T) * non_zero_num);
  out->SetMember(non_zero_crows, non_zero_cols, non_zero_elements, x_dims);
}

template <typename T, typename Context>
void SparseCooToDenseKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            DenseTensor* out) {
  const auto non_zero_num = x.nnz();
  const auto dense_dims = x.dims();
  const auto indices = x.non_zero_indices();
  const auto values = x.non_zero_elements();
  const auto indices_dims = indices.dims();
  int64_t sparse_dim = indices_dims[0];
  if (indices_dims.size() == 1) {
    sparse_dim = 1;
  }
  const int64_t dense_dim = values.dims().size() - 1;

  const T* x_data = values.data<T>();
  *out = phi::Empty(
      dev_ctx,
      DenseTensorMeta(x.dtype(), x.dims(), x.non_zero_elements().layout()));
  T* out_data = out->data<T>();
  int64_t base_offset = 1;
  for (int64_t i = 0; i < dense_dim; i++) {
    base_offset *= dense_dims[sparse_dim + i];
  }
  std::vector<int64_t> sparse_offsets(sparse_dim);
  int64_t offset = 1;
  for (int i = sparse_dim - 1; i >= 0; i--) {
    sparse_offsets[i] = offset;
    offset *= dense_dims[i];
  }

  memset(out_data, 0, sizeof(T) * out->numel());
  for (auto i = 0; i < non_zero_num; i++) {
    int64_t index = 0;
    for (int j = 0; j < sparse_dim; j++) {
      index +=
          indices.data<int64_t>()[j * non_zero_num + i] * sparse_offsets[j];
    }

    for (int j = 0; j < base_offset; j++) {
      out_data[index * base_offset + j] = x_data[i * base_offset + j];
    }
  }
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(dense_to_sparse_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::DenseToSparseCooKernel,
                   float,
                   double,
                   paddle::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(sparse_csr_to_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCsrToCooKernel,
                   float,
                   double,
                   paddle::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(sparse_coo_to_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooToCsrKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(dense_to_sparse_csr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::DenseToSparseCsrKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(sparse_coo_to_dense,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooToDenseKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(sparse_csr_to_dense,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCsrToDenseKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(coo_values,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CooValuesKernel,
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

PD_REGISTER_KERNEL(csr_values,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::CsrValuesKernel,
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
                   int64_t) {}
