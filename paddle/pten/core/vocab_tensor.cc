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

#include "paddle/pten/core/vocab_tensor.h"

#include "paddle/pten/core/compat/convert_utils.h"

namespace pten {
VocabTensor::VocabTensor(const std::unordered_map<std::string, int32_t>& data) {
  if (data.size() != 0) {
    data_ = data;
  }
  DDim tmp = DDim({static_cast<int64_t>(data_.size()), 1});
  dim_ = &tmp;
  cpu_place_ = paddle::platform::CPUPlace();
}

/// \brief Because vocab tensor is a resource handle, we provide a default
/// move constructor to support move semantics.
VocabTensor::VocabTensor(VocabTensor&& other) {
  if (other.numel() > 0) {
    data_ = std::move(other.data());
    DDim tmp = DDim({static_cast<int64_t>(data_.size()), 1});
    dim_ = &tmp;
    cpu_place_ = pten::CPUPlace();
  }
}
/// \brief VocabTensor deep copy constructor.
VocabTensor::VocabTensor(const VocabTensor& other) {
  if (other.numel() > 0) {
    data_ = other.data();
    DDim tmp = DDim({static_cast<int64_t>(data_.size()), 1});
    dim_ = &tmp;
    cpu_place_ = pten::CPUPlace();
  }
}

/// \brief VocabTensor deep copy assignment.
VocabTensor& VocabTensor::operator=(const VocabTensor& other) {
  if (other.numel() > 0) {
    data_ = other.data();
    DDim tmp = DDim({static_cast<int64_t>(data_.size()), 1});
    dim_ = &tmp;
    cpu_place_ = pten::CPUPlace();
  }
  return *this;
}

/// \brief VocabTensor shallow copy assignment.
VocabTensor& VocabTensor::operator=(VocabTensor&& other) {
  if (other.numel() > 0) {
    data_ = std::move(other.data());
    DDim tmp = DDim({static_cast<int64_t>(data_.size()), 1});
    dim_ = &tmp;
    cpu_place_ = pten::CPUPlace();
  }
  return *this;
}
}  // namespace pten
