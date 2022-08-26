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

void CheckTensor(const DenseTensor& t1, const DenseTensor& t2) {
  PADDLE_ENFORCE_EQ(
      t1.meta(),
      t2.meta(),
      phi::errors::PreconditionNotMet(
          "All tensor's meta must be same in TensorArray. "
          "But different tensors' meta are detected in TensorArray."));
}

TensorArray::TensorArray(const std::vector<DenseTensor>& vec) {
  if (vec.size() > 0) {
    for (auto tensor : vec) {
      CheckTensor(vec[0], tensor);
    }
    meta_ = vec[0].meta();
    place_ = vec[0].place();
  }
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

/// \brief Allocate memory with requested size for all tensors from allocator.
/// \return Void pointer
void* TensorArray::AllocateFrom(Allocator* allocator,
                                DataType dtype,
                                size_t requested_size) {
  for (auto tensor : tensors_) {
    tensor.AllocateFrom(allocator, dtype, requested_size);
  }
  return nullptr;
}

void TensorArray::push_back(const DenseTensor& tensor) {
  if (!empty()) {
    CheckTensor(tensors_[0], tensor);
  } else {
    meta_ = tensor.meta();
    if (tensor.IsInitialized()) {
      place_ = tensor.place();
    }
  }
  tensors_.push_back(tensor);
}

void TensorArray::emplace_back(const DenseTensor& tensor) {
  if (!empty()) {
    CheckTensor(tensors_[0], tensor);
  } else {
    meta_ = tensor.meta();
    if (tensor.IsInitialized()) {
      place_ = tensor.place();
    }
  }
  tensors_.emplace_back(tensor);
}

void TensorArray::emplace_back() {
  DenseTensor t;
  t.set_meta(meta_);
  tensors_.emplace_back(t);
}

}  // namespace phi
