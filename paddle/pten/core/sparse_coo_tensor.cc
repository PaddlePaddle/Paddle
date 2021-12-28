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

#include "paddle/pten/core/sparse_coo_tensor.h"

namespace pten {

SparseCooTensor::SparseCooTensor(const std::shared_ptr<Allocator>& a,
                                 const DenseTensorMeta& dense_meta) {
  // this->sparse_dim_ = 1;
  // this->dense_dim_ = 0;
  // auto indices_dims = paddle::framework::make_ddim({this->sparse_dim_, 1});
  // auto values_dims = paddle::framework::make_ddim({1});
  // DenseTensorMeta indices_meta(DataType::INT64, indices_dims,
  // DataLayout::ANY);
  // DenseTensorMeta values_meta(dense_meta.dtype, values_dims,
  // dense_meta.layout);
  // std::unique_ptr<DenseTensor> indices_ptr(new DenseTensor(a, indices_meta));
  // std::unique_ptr<DenseTensor> values_ptr(new DenseTensor(a, values_meta));
  // this->non_zero_indices_.reset(indices_ptr.release());
  // this->non_zero_elements_.reset(values_ptr.release());
}

SparseCooTensor::SparseCooTensor(const DenseTensor& non_zero_indices,
                                 const DenseTensor& non_zero_elements,
                                 const DDim& dims)
    : non_zero_indices_(non_zero_indices),
      non_zero_elements_(non_zero_elements) {
  this->coalesced_ = false;
  this->sparse_dim_ = non_zero_indices.dims()[0];
  this->dense_dim_ =
      non_zero_elements.dims().size() == 1 ? 0 : non_zero_elements.dims()[1];
  this->dims_ = dims;
}

int64_t SparseCooTensor::nnz() const {
  const auto indices_dims = non_zero_indices_.dims();
  if (indices_dims.size() == 1) {
    return indices_dims[0];
  }
  return indices_dims[1];
}

void SparseCooTensor::SetMember(const DenseTensor& non_zero_indices,
                                const DenseTensor& non_zero_elements,
                                const DDim& dims) {
  // this->non_zero_indices_.reset(non_zero_indices.release());
  // this->non_zero_elements_.reset(non_zero_elements.release());
  // this->dims_ = dims;
}

void SparseCooTensor::Resize(const DDim& dense_dims,
                             const int64_t sparse_dim,
                             const int64_t non_zero_num) {
  DDim indices_dims = paddle::framework::make_ddim({sparse_dim, non_zero_num});
  auto dense_dim = dense_dims.size() - sparse_dim;
  DDim values_dims;
  if (dense_dim) {
    std::vector<int64_t> dense_dim_vec(dense_dim + 1);
    dense_dim_vec[0] = non_zero_num;
    memcpy(&dense_dim_vec[1],
           dense_dims.Get() + sparse_dim,
           dense_dim * sizeof(dense_dims[0]));
    values_dims = paddle::framework::make_ddim(dense_dim_vec);
  } else {
    values_dims = paddle::framework::make_ddim({non_zero_num});
  }

  this->non_zero_indices_.Resize(indices_dims);
  this->non_zero_elements_.Resize(values_dims);
}

int64_t SparseCooTensor::Index(const std::vector<int64_t>& indices) const {
  PADDLE_ENFORCE_EQ(paddle::platform::is_cpu_place(place()),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "this method only support CPU place."));
  PADDLE_ENFORCE_EQ(
      indices.size(),
      sparse_dim_,
      paddle::platform::errors::InvalidArgument(
          "the query indices.size()=%d should be equal the sparse_dim=%d",
          static_cast<int>(indices.size()),
          static_cast<int>(sparse_dim_)));

  const int64_t non_zero_num = this->nnz();
  const int64_t* non_zero_indices = non_zero_indices_.data<int64_t>();
  for (int64_t i = 0; i < non_zero_num; i++) {
    bool flag = true;
    for (int64_t j = 0; j < sparse_dim_; j++) {
      if (non_zero_indices[j * non_zero_num + i] != indices[j]) {
        flag = false;
        break;
      }
    }
    if (flag == true) return i;
  }
  PADDLE_THROW(paddle::platform::errors::NotFound(
      "Input query indices is not in current rows table."));
}

int64_t SparseCooTensor::Index(int64_t key) const {
  PADDLE_ENFORCE_EQ(paddle::platform::is_cpu_place(place()),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "this method only support CPU place."));
  PADDLE_ENFORCE_EQ(sparse_dim_,
                    1,
                    paddle::platform::errors::InvalidArgument(
                        "this method is only valid when sparse_dim_ = 1"));
  const int64_t* indices = non_zero_indices_.data<int64_t>();
  for (int64_t i = 0; i < this->nnz(); i++) {
    if (indices[i] == key) {
      return i;
    }
  }
  PADDLE_THROW(paddle::platform::errors::NotFound(
      "Input id (%lld) is not in current rows table.", key));
}

}  // namespace pten
