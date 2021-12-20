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

#include "paddle/pten/core/sparse_csr_tensor.h"

namespace pten {

#define Check(non_zero_crows, non_zero_cols, non_zero_elements, dims)          \
  {                                                                            \
    PADDLE_ENFORCE_EQ(dims.size(),                                             \
                      2,                                                       \
                      paddle::platform::errors::InvalidArgument(               \
                          "the SparseCsrTensor only support 2-D Tensor."));    \
    PADDLE_ENFORCE_EQ(non_zero_crows->dtype(),                                 \
                      DataType::INT64,                                         \
                      paddle::platform::errors::InvalidArgument(               \
                          "the dtype of non_zero_crows should be int64_t."));  \
    PADDLE_ENFORCE_EQ(non_zero_cols->dtype(),                                  \
                      DataType::INT64,                                         \
                      paddle::platform::errors::InvalidArgument(               \
                          "the dtype of non_zero_cols should be int64_t."));   \
    PADDLE_ENFORCE_EQ(                                                         \
        non_zero_cols->place(),                                                \
        non_zero_crows->place(),                                               \
        paddle::platform::errors::InvalidArgument(                             \
            "non_zero_crows and non_zero_cols must have the same place."));    \
    PADDLE_ENFORCE_EQ(                                                         \
        non_zero_cols->place(),                                                \
        non_zero_elements->place(),                                            \
        paddle::platform::errors::InvalidArgument(                             \
            "non_zero_cols and non_zero_elements must have the same place.")); \
  }

SparseCsrTensor::SparseCsrTensor(std::unique_ptr<DenseTensor> non_zero_crows,
                                 std::unique_ptr<DenseTensor> non_zero_cols,
                                 std::unique_ptr<DenseTensor> non_zero_elements,
                                 const DDim& dims) {
  Check(non_zero_crows, non_zero_cols, non_zero_elements, dims);
  this->dims_ = dims;
  this->non_zero_crows_.reset(non_zero_crows.release());
  this->non_zero_cols_.reset(non_zero_cols.release());
  this->non_zero_elements_.reset(non_zero_elements.release());
}

int64_t SparseCsrTensor::nnz() const { return non_zero_elements_->dims()[0]; }

void SparseCsrTensor::SetMemberTensor(
    std::unique_ptr<DenseTensor> non_zero_crows,
    std::unique_ptr<DenseTensor> non_zero_cols,
    std::unique_ptr<DenseTensor> non_zero_elements,
    const DDim& dims) {
  Check(non_zero_crows, non_zero_cols, non_zero_elements, dims);
  this->dims_ = dims;
  this->non_zero_crows_.reset(non_zero_crows.release());
  this->non_zero_cols_.reset(non_zero_cols.release());
  this->non_zero_elements_.reset(non_zero_elements.release());
}

void SparseCsrTensor::Resize(const DDim& dims, const int64_t non_zero_num) {}

void SparseCsrTensor::Resize(const std::shared_ptr<Allocator>& a,
                             const DenseTensorMeta& meta,
                             const int64_t non_zero_num) {
  // non_zero_crows_->set_default_allocator(a);
  // non_zero_cols_->set_default_allocator(a);
  // non_zero_elements_->set_default_allocator(a);
}

int64_t* SparseCsrTensor::mutable_non_zero_crows() {
  return non_zero_crows_->mutable_data<int64_t>();
}

int64_t* SparseCsrTensor::mutable_non_zero_cols() {
  return non_zero_cols_->mutable_data<int64_t>();
}

template <typename T>
T* SparseCsrTensor::mutable_non_zero_elements() {
  return non_zero_elements_->mutable_data<int64_t>();
}

}  // namespace pten
