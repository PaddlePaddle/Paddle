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

void SparseCooTensor::Resize(const DDim& dense_dims,
                             const int64_t sparse_dim,
                             const int64_t non_zero_num) {
  PADDLE_ENFORCE_GE(non_zero_num,
                    this->nnz(),
                    phi::errors::InvalidArgument(
                        "the non_zero_num must be greater than or equal to the "
                        "origin non_zero_num."));
  PADDLE_ENFORCE_GE(sparse_dim,
                    1,
                    phi::errors::InvalidArgument(
                        "the sparse_dim must be greater than or equal 1."));
  PADDLE_ENFORCE_LE(
      sparse_dim,
      dense_dims.size(),
      phi::errors::InvalidArgument(
          "the sparse_dim must be less than or equal dense_dims."));

  DDim indices_dims = phi::make_ddim({sparse_dim, non_zero_num});
  auto dense_dim = dense_dims.size() - sparse_dim;
  DDim values_dims;
  if (dense_dim) {
    std::vector<int64_t> dense_dim_vec(dense_dim + 1);
    dense_dim_vec[0] = non_zero_num;
    memcpy(&dense_dim_vec[1],
           dense_dims.Get() + sparse_dim,
           dense_dim * sizeof(dense_dims[0]));
    values_dims = phi::make_ddim(dense_dim_vec);
  } else {
    values_dims = phi::make_ddim({non_zero_num});
  }

  this->non_zero_indices_.Resize(indices_dims);
  this->non_zero_elements_.Resize(values_dims);
}

void SparseCooTensor::SetMember(const DenseTensor& non_zero_indices,
                                const DenseTensor& non_zero_elements,
                                const DDim& dims,
                                const bool coalesced) {
  this->non_zero_indices_ = non_zero_indices;
  this->non_zero_elements_ = non_zero_elements;
  this->dims_ = dims;
  this->coalesced_ = coalesced;
}

}  // namespace phi
