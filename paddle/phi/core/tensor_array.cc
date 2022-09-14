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

namespace phi {

TensorArray::TensorArray(const std::vector<DenseTensor>& vec) {
  tensors_ = vec;
}

/// \brief Test whether the tensor's storage in TensorArray is allocated.
/// return Whether all tensors in TensorArray is allocated.
bool TensorArray::initialized() const {
  bool init = true;
  for (auto tensor : tensors_) {
    if (!tensor.IsInitialized()) {
      init = false;
    }
  }
  return init;
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
  PADDLE_THROW(errors::Unavailable("place() can't be used in TensorArray"));
  return tensors_[0].place();
}

DataType TensorArray::dtype() const {
  PADDLE_THROW(errors::Unavailable("dtype() can't be used in TensorArray"));
  return DataType::UNDEFINED;
}

DataLayout TensorArray::layout() const {
  PADDLE_THROW(errors::Unavailable("layout() can't be used in TensorArray"));
  return DataLayout::UNDEFINED;
}

bool TensorArray::valid() const {
  PADDLE_THROW(errors::Unavailable("valid() can't be used in TensorArray"));
  return false;
}

/// \brief Allocate memory with requested size for all tensors from allocator.
/// \return Void pointer
void* TensorArray::AllocateFrom(Allocator* allocator,
                                DataType dtype,
                                size_t requested_size) {
  for (size_t i = 0; i < tensors_.size(); i++) {
    tensors_[i].AllocateFrom(allocator, tensors_[i].dtype(), requested_size);
  }
  return nullptr;
}

void TensorArray::push_back(const DenseTensor& tensor) {
  tensors_.push_back(tensor);
}

void TensorArray::emplace_back(const DenseTensor& tensor) {
  tensors_.emplace_back(tensor);
}

void TensorArray::emplace_back() {
  DenseTensor t;
  tensors_.emplace_back(t);
}

}  // namespace phi
