// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/fleet_executor/dist_model_tensor_wrapper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

void DistModelDataBuf::Reset(void* data, size_t length) {
  Free();
  memory_owned_ = false;
  data_ = data;
  length_ = length;
}

void DistModelDataBuf::Free() {
  if (memory_owned_ && data_) {
    PADDLE_ENFORCE_GT(length_, 0UL,
                      platform::errors::PreconditionNotMet(
                          "Error occurred when deconstruct DistModelDataBuf: "
                          "it contains no data!"));
    // NOTE: if own the memory, it must be char* type
    delete[] static_cast<char*>(data_);
    data_ = nullptr;
    length_ = 0;
  }
}

void DistModelDataBuf::Resize(size_t length) {
  if (length_ >= length) {
    return;
  }
  if (memory_owned_) {
    Free();
    data_ = new char[length];
    length_ = length;
    memory_owned_ = true;
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "The memory is allocated externally, can not Resized"));
  }
}

DistModelDataBuf& DistModelDataBuf::operator=(const DistModelDataBuf& other) {
  if (!other.memory_owned_) {
    data_ = other.data_;
    length_ = other.length_;
    memory_owned_ = other.memory_owned_;
  } else {
    Resize(other.length_);
    if (other.length() && other.data()) {
      std::memcpy(data_, other.data(), other.length());
    } else if (other.length()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Invalid argument, null pointer data with length %u is passed",
          other.length()));
    }
    length_ = other.length_;
    memory_owned_ = true;
  }
  return *this;
}

DistModelDataBuf& DistModelDataBuf::operator=(DistModelDataBuf&& other) {
  data_ = other.data_;
  memory_owned_ = other.memory_owned_;
  length_ = other.length_;
  other.data_ = nullptr;
  other.length_ = 0;
  other.memory_owned_ = false;
  return *this;
}

DistModelDataBuf::DistModelDataBuf(DistModelDataBuf&& other)
    : data_(other.data_),
      length_(other.length_),
      memory_owned_(other.memory_owned_) {
  other.memory_owned_ = false;
  other.data_ = nullptr;
  other.length_ = 0;
}

DistModelDataBuf::DistModelDataBuf(const DistModelDataBuf& other) {
  *this = other;
}

}  // namespace distributed
}  // namespace paddle
