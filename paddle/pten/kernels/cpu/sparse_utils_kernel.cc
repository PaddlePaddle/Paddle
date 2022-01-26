/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/kernels/sparse_utils_kernel.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/backends/gpu/gpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

template <typename T>
inline bool is_zero(const T* data, const size_t n) {
  const T zero = static_cast<T>(0);
  for (size_t i = 0; i < n; i++) {
    if (data[i] != zero) {
      return false;
    }
  }
  return true;
}

template <typename T>
inline int64_t get_non_zero_num(const DenseTensor& dense,
                                const int64_t sparse_dim) {
  const auto& dims = dense.dims();
  PADDLE_ENFORCE_GE(
      dims.size(),
      sparse_dim,
      paddle::platform::errors::InvalidArgument(
          "sparse_dim(%d) should be less than or equal to dense.dim(%d)",
          sparse_dim,
          dims.size()));

  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];

  const T* data = dense.data<T>();
  int64_t non_zero_num = 0;
#pragma omp parallel for reduction(+ : non_zero_num)
  for (int64_t i = 0; i < rows; i++) {
    if (!is_zero(data + i * cols, cols)) {
      non_zero_num = non_zero_num + 1;
    }
  }
  return non_zero_num;
}

template <typename T, typename Context>
void DenseToSparseCooKernel(const Context& dev_ctx,
                            const DenseTensor& src,
                            const int64_t sparse_dim,
                            SparseCooTensor* dst) {
  const T* src_data = src.data<T>();
  const auto& src_dims = src.dims();

  int64_t non_zero_num = get_non_zero_num<T>(src, sparse_dim);

  const auto place = dev_ctx.GetPlace();
  const auto values_dims = InferDenseDims(src_dims, sparse_dim, non_zero_num);
  DenseTensorMeta indices_meta(
      DataType::INT64,
      paddle::framework::make_ddim(
          {sparse_dim, static_cast<int64_t>(non_zero_num)}),
      DataLayout::NCHW);
  DenseTensorMeta values_meta(src.meta().dtype, values_dims, src.meta().layout);
  pten::DenseTensor indices(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(indices_meta));
  pten::DenseTensor values(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          dev_ctx.GetPlace()),
      std::move(values_meta));
  int64_t* indices_data = indices.mutable_data<int64_t>(place);
  T* values_data = values.mutable_data<T>(place);

  auto dims_2d = flatten_to_2d(src_dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];

  int index = 0;
  for (int i = 0; i < rows; i++) {
    if (!is_zero(src_data + i * cols, cols)) {
      int64_t sparse_index = i;
      for (int64_t j = sparse_dim - 1; j >= 0; j--) {
        indices_data[j * non_zero_num + index] = sparse_index % src_dims[j];
        sparse_index /= src_dims[j];
      }
      memcpy(values_data + index * cols, src_data + i * cols, cols * sizeof(T));
      ++index;
    }
  }
  dst->SetMember(indices, values, src_dims, true);
}

template <typename T, typename Context>
void SparseCooToDenseKernel(const Context& dev_ctx,
                            const SparseCooTensor& src,
                            DenseTensor* dst) {
  const auto non_zero_num = src.nnz();
  const auto dense_dims = src.dims();
  const auto indices = src.non_zero_indices();
  const auto values = src.non_zero_elements();
  const auto indices_dims = indices.dims();
  int64_t sparse_dim = indices_dims[0];
  if (indices_dims.size() == 1) {
    sparse_dim = 1;
  }
  const int64_t dense_dim = values.dims().size() - 1;

  const auto place = dev_ctx.GetPlace();
  const T* src_data = values.data<T>();
  T* dst_data = dst->mutable_data<T>(place);
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

  memset(dst_data, 0, sizeof(T) * dst->numel());
  for (auto i = 0; i < non_zero_num; i++) {
    int64_t index = 0;
    for (int j = 0; j < sparse_dim; j++) {
      index +=
          indices.data<int64_t>()[j * non_zero_num + i] * sparse_offsets[j];
    }

    for (int j = 0; j < base_offset; j++) {
      dst_data[index * base_offset + j] = src_data[i * base_offset + j];
    }
  }
}

}  // namespace pten

PT_REGISTER_KERNEL(dense_to_sparse_coo,
                   CPU,
                   ALL_LAYOUT,
                   pten::DenseToSparseCooKernel,
                   float,
                   double) {}

PT_REGISTER_KERNEL(sparse_coo_to_dense,
                   CPU,
                   ALL_LAYOUT,
                   pten::SparseCooToDenseKernel,
                   float,
                   double) {}
