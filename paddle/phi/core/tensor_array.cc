/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/tensor_array.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

TensorArray::TensorArray(const std::vector<DenseTensor>& vec) {
  tensors_ = vec;
}

/// \brief Test whether the holder is created.
/// \return Whether the holder is created.
bool TensorArray::has_allocation() const {
  if (tensors_.empty()) {
    return false;
  }

  for (auto const& tensor : tensors_) {
    if (!tensor.has_allocation()) {
      return false;
    }
  }
  return true;
}

/// \brief Test whether the tensor's storage in TensorArray is allocated.
/// return Whether all tensors in TensorArray is allocated.
bool TensorArray::initialized() const {
  if (tensors_.empty()) {
    return false;
  }

  for (auto const& tensor : tensors_) {
    if (!tensor.initialized()) {
      return false;
    }
  }
  return true;
}

int64_t TensorArray::numel() const {
  PADDLE_THROW(errors::Unavailable("numel() can't be used in TensorArray"));
  return -1;
}

const DDim& TensorArray::dims() const {
  PADDLE_THROW(errors::Unavailable("dims() can't be used in TensorArray"));
  return tensors_[0].dims();
}

const Place& TensorArray::place() const {
  PADDLE_ENFORCE_NE(
      tensors_.size(), 0, errors::Unavailable("TensorArray is not assigned."));

  const Place& place = tensors_[0].place();
  for (size_t i = 1; i < tensors_.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        tensors_[i].place(),
        place,
        errors::Unavailable(
            "The Place of all tensors in TensorArray must be consistent. The "
            "current place is %s, but the previous place is %s.",
            tensors_[i].place(),
            place));
  }
  return place;
}

DataType TensorArray::dtype() const { return dtype_; }

void TensorArray::set_type(const DataType dtype) {
  for (auto& tensor : tensors_) {
    tensor.set_type(dtype);
  }
  dtype_ = dtype;
}

DataLayout TensorArray::layout() const { return layout_; }

void TensorArray::set_layout(DataLayout layout) {
  for (auto& tensor : tensors_) {
    tensor.set_layout(layout);
  }
  layout_ = layout;
}

bool TensorArray::valid() const {
  PADDLE_THROW(errors::Unavailable("valid() can't be used in TensorArray"));
  return false;
}

/// \brief Allocate memory with requested size for all tensors from allocator.
/// \return Void pointer
void* TensorArray::AllocateFrom(Allocator* allocator,
                                DataType dtype,
                                size_t requested_size,
                                bool fake_allc) {
  for (auto& tensor : tensors_) {
    tensor.AllocateFrom(allocator, tensor.dtype(), requested_size, fake_allc);
  }
  return nullptr;
}

void TensorArray::push_back(const DenseTensor& tensor) {
  tensors_.push_back(tensor);
}

void TensorArray::pop(size_t i) {
  PADDLE_ENFORCE_LT(i,
                    tensors_.size(),
                    errors::OutOfRange("The size of TensorArray is %d, "
                                       "but the received index is %d.",
                                       tensors_.size(),
                                       i));
  tensors_.erase(tensors_.begin() + i);
}

void TensorArray::emplace_back(const DenseTensor& tensor) {
  tensors_.emplace_back(tensor);
}

void TensorArray::emplace_back() {
  DenseTensor t;
  tensors_.emplace_back(t);
}

}  // namespace phi
