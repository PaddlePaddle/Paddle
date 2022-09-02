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

#include "paddle/phi/core/sparse_coo_tensor.h"

namespace phi {

SparseCooTensor::SparseCooTensor() {
  DenseTensor non_zero_indices, non_zero_elements;
  this->SetMembers(non_zero_indices, non_zero_elements, {1}, true);
}

SparseCooTensor::SparseCooTensor(const DenseTensor& non_zero_indices,
                                 const DenseTensor& non_zero_elements,
                                 const DDim& dims)
    : non_zero_indices_(non_zero_indices),
      non_zero_elements_(non_zero_elements),
      coalesced_(false),
      dims_(dims) {}

SparseCooTensor::SparseCooTensor(DenseTensor&& non_zero_indices,
                                 DenseTensor&& non_zero_elements,
                                 const DDim& dims)
    : non_zero_indices_(non_zero_indices),
      non_zero_elements_(non_zero_elements),
      coalesced_(false),
      dims_(dims) {}

SparseCooTensor::SparseCooTensor(const SparseCooTensor& other)
    : non_zero_indices_(other.non_zero_indices_),
      non_zero_elements_(other.non_zero_elements_),
      dims_(other.dims_) {
  this->coalesced_ = other.coalesced_;
}

SparseCooTensor SparseCooTensor::operator=(const SparseCooTensor& other) {
  this->dims_ = other.dims_;
  this->non_zero_indices_ = other.non_zero_indices_;
  this->non_zero_elements_ = other.non_zero_elements_;
  this->coalesced_ = other.coalesced_;
  return *this;
}

void* SparseCooTensor::AllocateFrom(Allocator* allocator,
                                    DataType dtype,
                                    size_t requested_size) {
  return non_zero_elements_.AllocateFrom(allocator, dtype, requested_size);
  // // Is there need to add the following code ?
  // non_zero_indices_.AllocateFrom(allocator, dtype, requested_size);
}

int64_t SparseCooTensor::nnz() const {
  const auto indices_dims = non_zero_indices_.dims();
  if (indices_dims.size() == 0) {
    return 0;
  } else if (indices_dims.size() == 1) {
    return indices_dims[0];
  } else {
    return indices_dims[1];
  }
}

void SparseCooTensor::Resize(const DDim& original_dims,
                             const int64_t num_sparse_dims,
                             const int64_t num_non_zero) {
  PADDLE_ENFORCE_GE(num_non_zero,
                    this->nnz(),
                    phi::errors::InvalidArgument(
                        "the num_non_zero must be greater than or equal to the "
                        "original number of non zero elements."));
  PADDLE_ENFORCE_GE(num_sparse_dims,
                    1,
                    phi::errors::InvalidArgument(
                        "the num_sparse_dims must be greater than or equal to 1."));
  PADDLE_ENFORCE_LE(
      num_sparse_dims,
      original_dims.size(),
      phi::errors::InvalidArgument(
          "the num_sparse_dims must be less than or equal to the rank of dense_dims."));

  DDim indices_dims = phi::make_ddim({num_sparse_dims, num_non_zero});
  auto num_dense_dims = original_dims.size() - num_sparse_dims;
  DDim values_dims;
  if (num_dense_dims > 0) {
    std::vector<int64_t> dense_dim_vec(num_dense_dims + 1);
    dense_dim_vec[0] = num_non_zero;
    memcpy(&dense_dim_vec[1],
           original_dims.Get() + num_sparse_dims,
           num_dense_dims * sizeof(original_dims[0]));
    values_dims = phi::make_ddim(dense_dim_vec);
  } else {
    values_dims = phi::make_ddim({num_non_zero});
  }

  this->non_zero_indices_.Resize(indices_dims);
  this->non_zero_elements_.Resize(values_dims);
}

void SparseCooTensor::SetMembers(const DenseTensor& non_zero_indices,
                                const DenseTensor& non_zero_elements,
                                const DDim& dims,
                                const bool coalesced) {
  this->non_zero_indices_ = non_zero_indices;
  this->non_zero_elements_ = non_zero_elements;
  this->dims_ = dims;
  this->coalesced_ = coalesced;
}

int32_t SparseCooTensor::num_sparse_dims() const {
  return non_zero_indices_.dims()[0];
}

int32_t SparseCooTensor::num_dense_dims() const {
  return dims_.size() - num_sparse_dims();
}

}  // namespace phi
