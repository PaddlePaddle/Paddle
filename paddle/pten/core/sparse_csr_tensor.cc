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
    PADDLE_ENFORCE_EQ(non_zero_crows.dtype(),                                  \
                      DataType::INT64,                                         \
                      paddle::platform::errors::InvalidArgument(               \
                          "the dtype of non_zero_crows should be int64_t."));  \
    PADDLE_ENFORCE_EQ(non_zero_cols.dtype(),                                   \
                      DataType::INT64,                                         \
                      paddle::platform::errors::InvalidArgument(               \
                          "the dtype of non_zero_cols should be int64_t."));   \
    PADDLE_ENFORCE_EQ(                                                         \
        non_zero_cols.place(),                                                 \
        non_zero_crows.place(),                                                \
        paddle::platform::errors::InvalidArgument(                             \
            "non_zero_crows and non_zero_cols must have the same place."));    \
    PADDLE_ENFORCE_EQ(                                                         \
        non_zero_cols.place(),                                                 \
        non_zero_elements.place(),                                             \
        paddle::platform::errors::InvalidArgument(                             \
            "non_zero_cols and non_zero_elements must have the same place.")); \
  }

SparseCsrTensor::SparseCsrTensor(const std::shared_ptr<Allocator>& a,
                                 const DenseTensorMeta& meta) {}

SparseCsrTensor::SparseCsrTensor(const DenseTensor& non_zero_crows,
                                 const DenseTensor& non_zero_cols,
                                 const DenseTensor& non_zero_elements,
                                 const DDim& dims)
    : non_zero_crows_(non_zero_crows),
      non_zero_cols_(non_zero_cols),
      non_zero_elements_(non_zero_elements),
      dims_(dims) {
  Check(non_zero_crows_, non_zero_cols_, non_zero_elements_, dims_);
}

SparseCsrTensor::SparseCsrTensor(DenseTensor&& non_zero_crows,
                                 DenseTensor&& non_zero_cols,
                                 DenseTensor&& non_zero_elements,
                                 const DDim& dims)
    : non_zero_crows_(std::move(non_zero_crows)),
      non_zero_cols_(std::move(non_zero_cols)),
      non_zero_elements_(std::move(non_zero_elements)),
      dims_(dims) {
  Check(non_zero_crows_, non_zero_cols_, non_zero_elements_, dims_);
}

int64_t SparseCsrTensor::nnz() const { return non_zero_elements_.dims()[0]; }

void SparseCsrTensor::SetMemberTensor(const DenseTensor& non_zero_crows,
                                      const DenseTensor& non_zero_cols,
                                      const DenseTensor& non_zero_elements,
                                      const DDim& dims) {
  Check(non_zero_crows, non_zero_cols, non_zero_elements, dims);
  this->dims_ = dims;
  // need DenseTensor implementation operator=
  // this->non_zero_crows_ = non_zero_crows;
  // this->non_zero_cols_ = non_zero_cols;
  // this->non_zero_elements_ = non_zero_elements;
}

void SparseCsrTensor::Resize(const DDim& dense_dims,
                             const int64_t non_zero_num) {
  PADDLE_ENFORCE_EQ(dense_dims.size(),
                    2,
                    paddle::platform::errors::InvalidArgument(
                        "the SparseCsrTensor only support 2-D Tensor."));
  DDim crows_dims = paddle::framework::make_ddim({dense_dims[0] + 1});
  this->non_zero_crows_.Resize(crows_dims);

  DDim nnz_dims = paddle::framework::make_ddim({non_zero_num});
  this->non_zero_cols_.Resize(nnz_dims);

  this->non_zero_elements_.Resize(nnz_dims);
}

void SparseCsrTensor::Resize(const std::shared_ptr<Allocator>& a,
                             const DenseTensorMeta& meta,
                             const int64_t non_zero_num) {
  // non_zero_crows_->set_default_allocator(a);
  // non_zero_cols_->set_default_allocator(a);
  // non_zero_elements_->set_default_allocator(a);
}

}  // namespace pten
