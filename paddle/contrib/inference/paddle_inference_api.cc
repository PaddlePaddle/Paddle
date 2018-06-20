/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/contrib/inference/paddle_inference_api.h"

namespace paddle {

PaddleBuf::PaddleBuf(PaddleBuf&& other)
    : data_(other.data_),
      length_(other.length_),
      memory_owned_(other.memory_owned_) {
  other.memory_owned_ = false;
  other.data_ = nullptr;
  other.length_ = 0;
}

PaddleBuf::PaddleBuf(const PaddleBuf& other) { *this = other; }

PaddleBuf& PaddleBuf::operator=(const PaddleBuf& other) {
  // only the buffer with external memory can be copied
  assert(!other.memory_owned_);
  data_ = other.data_;
  length_ = other.length_;
  memory_owned_ = other.memory_owned_;
  return *this;
}

void PaddleBuf::Resize(size_t length) {
  // Only the owned memory can be reset, the external memory can't be changed.
  if (length_ == length) return;
  assert(memory_owned_);
  Free();
  data_ = new char[length];
  length_ = length;
  memory_owned_ = true;
}

void PaddleBuf::Reset(void* data, size_t length) {
  Free();
  memory_owned_ = false;
  data_ = data;
  length_ = length;
}

void PaddleBuf::Free() {
  if (memory_owned_ && data_) {
    assert(length_ > 0);
    delete static_cast<char*>(data_);
    data_ = nullptr;
    length_ = 0;
  }
}

}  // namespace paddle