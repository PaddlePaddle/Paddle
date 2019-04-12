// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdlib.h>
#include <sstream>
#include "paddle/fluid/framework/commit.h"
#include "paddle/fluid/inference/lite/paddle_api.h"

namespace paddle {

int PaddleDtypeSize(PaddleDType dtype) {
  switch (dtype) {
    case PaddleDType::FLOAT32:
      return sizeof(float);
    case PaddleDType::INT64:
      return sizeof(int64_t);
    case PaddleDType::INT32:
      return sizeof(int32_t);
    default:
      assert(false);
      return -1;
  }
}

PaddleBuf::PaddleBuf(PaddleBuf &&other)
    : data_(other.data_),
      length_(other.length_),
      memory_owned_(other.memory_owned_) {
  other.memory_owned_ = false;
  other.data_ = nullptr;
  other.length_ = 0;
}

PaddleBuf::PaddleBuf(const PaddleBuf &other) { *this = other; }

PaddleBuf &PaddleBuf::operator=(const PaddleBuf &other) {
  if (!other.memory_owned_) {
    data_ = other.data_;
    length_ = other.length_;
    memory_owned_ = other.memory_owned_;
  } else {
    Resize(other.length());
    memcpy(data_, other.data(), other.length());
    length_ = other.length();
    memory_owned_ = true;
  }
  return *this;
}

PaddleBuf &PaddleBuf::operator=(PaddleBuf &&other) {
  // only the buffer with external memory can be copied
  data_ = other.data_;
  length_ = other.length_;
  memory_owned_ = other.memory_owned_;
  other.data_ = nullptr;
  other.length_ = 0;
  other.memory_owned_ = false;
  return *this;
}

void PaddleBuf::Resize(size_t length) {
  // Only the owned memory can be reset, the external memory can't be changed.
  if (length_ >= length) return;
  if (memory_owned_) {
    Free();
    data_ = malloc(length);
    length_ = length;
    memory_owned_ = true;
  } else {
    // PADDLE_THROW("The memory is allocated externally, can not Resized");
  }
}

void PaddleBuf::Reset(void *data, size_t length) {
  Free();
  memory_owned_ = false;
  data_ = data;
  length_ = length;
}

void PaddleBuf::Free() {
  if (memory_owned_ && data_) {
    // PADDLE_ENFORCE_GT(length_, 0UL);
    free(static_cast<char *>(data_));
    data_ = nullptr;
    length_ = 0;
  }
}

std::string get_version() {
  std::stringstream ss;
  ss << "version: " << framework::paddle_version() << "\n";
  ss << "commit: " << framework::paddle_commit() << "\n";
  ss << "branch: " << framework::paddle_compile_branch() << "\n";
  return ss.str();
}

}  // namespace paddle
