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

#include "paddle/pten/kernels/sparse/cpu/sparse_coo_tensor_util.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/core/tensor_meta.h"
#include "paddle/pten/kernels/hybird/sparse/cpu/sparse_utils.h"

namespace pten {

template <typename T>
void ToSparseCoo(const CPUContext& dev_ctx,
                 const DenseTensor& src,
                 const int64_t sparse_dim,
                 SparseCooTensor* dst) {
  const T* src_data = src.data<T>();
  const auto& src_dims = src.dims();

  int64_t non_zero_num = get_non_zero_num<T>(src, sparse_dim);

  // auto dense_dim = src_dims.size() - sparse_dim;
  // auto indices_dims = paddle::framework::make_ddim({sparse_dim,
  // non_zero_num});
  dst->Resize(src_dims, sparse_dim, non_zero_num);
  // DDim values_dims;
  // if (dense_dim) {
  //  std::vector<int64_t> dense_dims(dense_dim + 1);
  //  dense_dims[0] = non_zero_num;
  //  memcpy(&dense_dims[1],
  //         src_dims.Get() + sparse_dim,
  //         dense_dim * sizeof(src_dims[0]));
  //  values_dims = paddle::framework::make_ddim(dense_dims);
  //} else {
  //  values_dims = paddle::framework::make_ddim({non_zero_num});
  //}

  // const auto allocator =
  //    std::make_shared<paddle::experimental::DefaultAllocator>(src.place());
  // DenseTensorMeta indices_meta(DataType::INT64, indices_dims,
  // DataLayout::NCHW);
  // DenseTensorMeta values_meta(src.meta().dtype, values_dims,
  // src.meta().layout);
  // std::unique_ptr<DenseTensor> indices_ptr(
  //    new DenseTensor(allocator, indices_meta));
  // std::unique_ptr<DenseTensor> values_ptr(
  //    new DenseTensor(allocator, values_meta));

  // int64_t* indices_data = indices_ptr->mutable_data<int64_t>();
  // T* values_data = values_ptr->mutable_data<T>();

  int64_t* indices_data = dst->mutable_non_zero_indices();
  T* values_data = dst->mutable_non_zero_elements<T>();

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
}

}  // namespace pten

PT_REGISTER_KERNEL(
    to_sparse_coo, CPU, ALL_LAYOUT, pten::ToSparseCoo, float, double) {}
