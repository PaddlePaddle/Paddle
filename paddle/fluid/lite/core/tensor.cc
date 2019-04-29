// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/lite/core/tensor.h"

namespace paddle {
namespace lite {

std::ostream &operator<<(std::ostream &os, const DDim &dims) {
  if (dims.empty()) {
    os << "[]";
    return os;
  }

  os << "[";
  for (size_t i = 0; i < dims.size() - 1; i++) {
    os << dims[i] << " ";
  }
  os << dims.back() << "]";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
  os << "Tensor:" << '\n';
  os << "dim: " << tensor.dims() << '\n';
  for (int i = 0; i < product(tensor.dims()); i++) {
    os << tensor.data<float>()[i] << " ";
  }
  os << "\n";
  return os;
}

void Tensor::ShareDataWith(const Tensor &other) {
  buffer_ = other.buffer_;
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
}

void *Tensor::mutable_data(size_t memory_size) {
  buffer_->ResetLazy(target_, memory_size);
  return buffer_->data();
}

void *Tensor::mutable_data(TargetType target, size_t memory_size) {
  target_ = target;
  return mutable_data(memory_size);
}

void Tensor::CopyDataFrom(const Tensor &other) {
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
  buffer_->CopyDataFrom(*other.buffer_, memory_size_);
}

}  // namespace lite
}  // namespace paddle
