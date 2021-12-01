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

namespace pten {

template <typename T>
void ToSparseCoo(const CPUContext& dev_ctx,
                 const DenseTensor& src,
                 const int64_t sparse_dim,
                 SparseCooTensor* dst) {
  const T* src_data = src.data<T>();
  const auto& src_dims = src.dims();
  // TODO(zhangkaihuo) use an API/kernel/function to implement this function
  int64_t non_zero_num = std::accumulate(
      src_data, src_data + src.numel(), 0, [&](int non_zero_num, const T& a) {
        if (a) {
          return non_zero_num + 1;
        } else {
          return non_zero_num;
        }
      });

  auto dense_dim = src.dims().size() - sparse_dim;
  auto indices_dims = paddle::framework::make_ddim({sparse_dim, non_zero_num});
  DDim values_dims;
  if (dense_dim) {
    values_dims = paddle::framework::make_ddim({non_zero_num, dense_dim});
  } else {
    values_dims = paddle::framework::make_ddim({non_zero_num});
  }

  const auto allocator =
      std::make_shared<paddle::experimental::DefaultAllocator>(src.place());
  DenseTensorMeta indices_meta(DataType::INT64, indices_dims, DataLayout::ANY);
  DenseTensorMeta values_meta(src.meta().dtype, values_dims, src.meta().layout);
  std::unique_ptr<DenseTensor> indices_ptr(
      new DenseTensor(allocator, indices_meta));
  std::unique_ptr<DenseTensor> values_ptr(
      new DenseTensor(allocator, values_meta));

  int64_t* indices_data = indices_ptr->mutable_data<int64_t>();
  T* values_data = values_ptr->mutable_data<T>();

  // 2-D
  int index = 0;
  for (int i = 0; i < src_dims[0]; i++) {
    for (int j = 0; j < src_dims[1]; j++) {
      T value = src_data[i * src_dims[1] + j];
      if (value) {
        indices_data[index] = i;
        indices_data[non_zero_num + index] = j;
        values_data[index] = value;
        ++index;
      }
    }
  }

  dst->set_indices_and_values_unsafe(
      std::move(indices_ptr), std::move(values_ptr), src.dims());
}

}  // namespace pten

PT_REGISTER_MODULE(SparseCooTensorUtilCPU);

PT_REGISTER_KERNEL(
    "to_sparse_coo", CPU, ANY, pten::ToSparseCoo, float, double) {}
