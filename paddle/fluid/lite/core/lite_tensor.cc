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

#include "paddle/fluid/lite/core/lite_tensor.h"

namespace paddle {
namespace lite {

void TensorLite::ShareDataWith(const TensorLite &other) {
  buffer_ = other.buffer_;
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
}

void *TensorLite::mutable_data(size_t memory_size) {
  buffer_->ResetLazy(target_, memory_size);
  return buffer_->data();
}

void *TensorLite::mutable_data(TargetType target, size_t memory_size) {
  target_ = target;
  return mutable_data(memory_size);
}

void TensorLite::CopyDataFrom(const TensorLite &other) {
  dims_ = other.dims_;
  target_ = other.target_;
  lod_ = other.lod_;
  memory_size_ = other.memory_size_;
  buffer_->CopyDataFrom(*other.buffer_, memory_size_);
}

}  // namespace lite
}  // namespace paddle
