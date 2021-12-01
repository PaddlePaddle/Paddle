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
  this->sparse_dim_ = 1;
  this->dense_dim_ = 0;
  auto indices_dims = paddle::framework::make_ddim({this->sparse_dim_, 1});
  auto values_dims = paddle::framework::make_ddim({1});
  DenseTensorMeta indices_meta(DataType::INT64, indices_dims, DataLayout::ANY);
  DenseTensorMeta values_meta(dense_meta.dtype, values_dims, dense_meta.layout);
  std::unique_ptr<DenseTensor> indices_ptr(new DenseTensor(a, indices_meta));
  std::unique_ptr<DenseTensor> values_ptr(new DenseTensor(a, values_meta));
  this->indices_.reset(indices_ptr.release());
  this->values_.reset(values_ptr.release());
}

SparseCooTensor::SparseCooTensor(std::unique_ptr<DenseTensor> indices,
                                 std::unique_ptr<DenseTensor> values,
                                 const DDim& dims) {
  this->coalesced_ = false;
  this->sparse_dim_ = indices->dims()[0];
  this->dense_dim_ = values->dims().size() == 1 ? 0 : values->dims()[1];
  this->dims_ = dims;
  this->indices_.reset(indices.release());
  this->values_.reset(values.release());
}

void SparseCooTensor::set_indices_and_values_unsafe(
    std::unique_ptr<DenseTensor> indices,
    std::unique_ptr<DenseTensor> values,
    const DDim& dims) {
  this->indices_.reset(indices.release());
  this->values_.reset(values.release());
  this->dims_ = dims;
}
}  // namespace pten
