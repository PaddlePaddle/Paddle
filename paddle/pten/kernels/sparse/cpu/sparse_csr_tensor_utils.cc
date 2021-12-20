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

#include "paddle/pten/kernels/sparse/cpu/sparse_csr_tensor_utils.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/kernels/hybird/sparse/cpu/sparse_utils.h"

namespace pten {

template <typename T>
void ToSparseCsr(const CPUContext& dev_ctx,
                 const DenseTensor& src,
                 SparseCsrTensor* dst) {
  PADDLE_ENFORCE_EQ(src.dims().size(),
                    2,
                    paddle::platform::errors::InvalidArgument(
                        "SparseCsrTensor only support 2-D Tensor."));

  const T* src_data = src.data<T>();
  const auto& src_dims = src.dims();

  int64_t non_zero_num = get_non_zero_num<T>(src, 2);

  auto non_zero_dims = paddle::framework::make_ddim({non_zero_num});
  auto crows_dims = paddle::framework::make_ddim({src_dims[0] + 1});
  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(src.place());
  DenseTensorMeta crows_meta(DataType::INT64, crows_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> crows_ptr(
      new DenseTensor(allocator, crows_meta));
  DenseTensorMeta cols_meta(DataType::INT64, non_zero_dims, DataLayout::ANY);
  std::unique_ptr<DenseTensor> cols_ptr(new DenseTensor(allocator, cols_meta));
  DenseTensorMeta values_meta(src.dtype(), non_zero_dims, src.layout());
  std::unique_ptr<DenseTensor> values_ptr(
      new DenseTensor(allocator, values_meta));

  int64_t* crows_data = crows_ptr->mutable_data<int64_t>();
  int64_t* cols_data = cols_ptr->mutable_data<int64_t>();
  T* values_data = values_ptr->mutable_data<T>();

  int non_zero_count = 0;
  for (int i = 0; i < src_dims[0]; i++) {
    crows_data[i] = non_zero_count;
    for (int j = 0; j < src_dims[1]; j++) {
      const T data = src_data[i * src_dims[1] + j];
      if (data != static_cast<T>(0)) {
        cols_data[non_zero_count] = j;
        values_data[non_zero_count] = data;
        ++non_zero_count;
      }
    }
  }
  crows_data[src_dims[0]] = non_zero_count;
  dst->SetMemberTensor(std::move(crows_ptr),
                       std::move(cols_ptr),
                       std::move(values_ptr),
                       src_dims);
}

template <typename T>
void SparseCsrToDense(const CPUContext& dev_ctx,
                      const SparseCsrTensor& src,
                      DenseTensor* dst) {
  const auto& non_zero_crows = src.non_zero_crows();
  const auto& non_zero_cols = src.non_zero_cols();
  const auto& non_zero_elements = src.non_zero_elements();

  const auto& dense_dims = src.dims();

  T* out_data = dst->mutable_data<T>();
  memset(out_data, 0, sizeof(T) * dst->numel());

  const auto rows = non_zero_crows.numel() - 1;
  const auto cols = dense_dims[1];
  const int64_t* crows_data = non_zero_crows.data<int64_t>();
  const int64_t* cols_data = non_zero_cols.data<int64_t>();
  const T* elements_data = non_zero_elements.data<T>();
  int non_zero_num = 0;
  for (auto row = 0; row < rows; row++) {
    for (int64_t i = crows_data[row]; i < crows_data[row + 1]; i++) {
      out_data[row * cols + cols_data[non_zero_num]] =
          elements_data[non_zero_num];
      ++non_zero_num;
    }
  }
}

}  // namespace pten

PT_REGISTER_KERNEL(to_sparse_csr, CPU, ANY, pten::ToSparseCsr, float, double) {}
PT_REGISTER_KERNEL(
    sparse_csr_to_dense, CPU, ANY, pten::SparseCsrToDense, float, double) {}
