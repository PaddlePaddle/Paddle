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

#include "paddle/phi/core/sparse_csr_tensor.h"

namespace phi {

inline void check_shape(const DDim& dims) {
  bool valid = dims.size() == 2 || dims.size() == 3;

  PADDLE_ENFORCE(valid,
                 phi::errors::InvalidArgument(
                     "the SparseCsrTensor only support 2-D Tensor."));
}
#define Check(non_zero_crows, non_zero_cols, non_zero_elements, dims)          \
  {                                                                            \
    check_shape(dims);                                                         \
    PADDLE_ENFORCE_EQ(                                                         \
        non_zero_cols.place(),                                                 \
        non_zero_crows.place(),                                                \
        phi::errors::InvalidArgument(                                          \
            "non_zero_crows and non_zero_cols must have the same place."));    \
    PADDLE_ENFORCE_EQ(                                                         \
        non_zero_cols.place(),                                                 \
        non_zero_elements.place(),                                             \
        phi::errors::InvalidArgument(                                          \
            "non_zero_cols and non_zero_elements must have the same place.")); \
  }

SparseCsrTensor::SparseCsrTensor(const DenseTensor& non_zero_crows,
                                 const DenseTensor& non_zero_cols,
                                 const DenseTensor& non_zero_elements,
                                 const DDim& dims)
    : non_zero_crows_(non_zero_crows),
      non_zero_cols_(non_zero_cols),
      non_zero_elements_(non_zero_elements),
      dims_(dims) {
  if (non_zero_crows.initialized()) {
    Check(non_zero_crows_, non_zero_cols_, non_zero_elements_, dims_);
  } else {
    // create a empty tensor
    check_shape(dims);
  }
}

SparseCsrTensor::SparseCsrTensor(const SparseCsrTensor& other)
    : non_zero_crows_(other.non_zero_crows_),
      non_zero_cols_(other.non_zero_cols_),
      non_zero_elements_(other.non_zero_elements_),
      dims_(other.dims_) {}

SparseCsrTensor& SparseCsrTensor::operator=(const SparseCsrTensor& other) {
  this->dims_ = other.dims();
  this->non_zero_crows_ = other.non_zero_crows();
  this->non_zero_cols_ = other.non_zero_cols();
  this->non_zero_elements_ = other.non_zero_elements();
  return *this;
}

void* SparseCsrTensor::AllocateFrom(Allocator* allocator,
                                    DataType dtype,
                                    size_t requested_size) {
  return non_zero_elements_.AllocateFrom(allocator, dtype, requested_size);
}

void SparseCsrTensor::Resize(const DDim& dense_dims,
                             const int64_t non_zero_num) {
  PADDLE_ENFORCE(this->initialized(),
                 phi::errors::InvalidArgument(
                     "the SparseCsrTensor must be initialized when call Resize "
                     "function."));
  check_shape(dense_dims);

  int64_t crows_size = dense_dims[0] + 1;
  if (dense_dims.size() == 3) {
    // batch_size = dims[0]
    crows_size = dense_dims[0] * (dense_dims[1] + 1);
  }

  DDim crows_dims = phi::make_ddim({crows_size});
  this->non_zero_crows_.Resize(crows_dims);

  DDim col_dims = phi::make_ddim({non_zero_num});
  this->non_zero_cols_.Resize(col_dims);
  this->non_zero_elements_.Resize(col_dims);
}

void SparseCsrTensor::SetMember(const DenseTensor& non_zero_crows,
                                const DenseTensor& non_zero_cols,
                                const DenseTensor& non_zero_elements,
                                const DDim& dims) {
  Check(non_zero_crows, non_zero_cols, non_zero_elements, dims);
  this->non_zero_crows_ = non_zero_crows;
  this->non_zero_cols_ = non_zero_cols;
  this->non_zero_elements_ = non_zero_elements;
  this->dims_ = dims;
}

}  // namespace phi
