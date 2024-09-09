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

#include <sstream>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/commit.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {

int PaddleDtypeSize(PaddleDType dtype) {
  switch (dtype) {
    case PaddleDType::FLOAT32:
      return sizeof(float);
    case PaddleDType::BFLOAT16:
      return sizeof(uint16_t);
    case PaddleDType::INT64:
      return sizeof(int64_t);
    case PaddleDType::INT32:
      return sizeof(int32_t);
    case PaddleDType::UINT8:
      return sizeof(uint8_t);
    default:
      assert(false);
      return -1;
  }
}

PaddleBuf::PaddleBuf(PaddleBuf &&other) noexcept
    : data_(other.data_),
      length_(other.length_),
      memory_owned_(other.memory_owned_) {
  other.memory_owned_ = false;
  other.data_ = nullptr;
  other.length_ = 0;
}

PaddleBuf::PaddleBuf(const PaddleBuf &other) { *this = other; }

PaddleBuf &PaddleBuf::operator=(const PaddleBuf &other) {
  if (this == &other) return *this;
  if (!other.memory_owned_) {
    data_ = other.data_;
    length_ = other.length_;
    memory_owned_ = other.memory_owned_;
  } else {
    Resize(other.length());
    // if other.length() == 0 or other.data() == nullptr, then the memcpy
    // behavior is undefined
    if (other.length() && other.data())
      memcpy(data_, other.data(), other.length());
    else if (other.length())
      PADDLE_THROW(common::errors::InvalidArgument(
          "Invalid argument, null pointer data with length %u is passed",
          other.length()));

    length_ = other.length();
    memory_owned_ = true;
  }
  return *this;
}

PaddleBuf &PaddleBuf::operator=(PaddleBuf &&other) noexcept {
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
    data_ = new char[length];
    length_ = length;
    memory_owned_ = true;
  } else {
    PADDLE_THROW(common::errors::PreconditionNotMet(
        "The memory is allocated externally, can not Resized"));
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
    PADDLE_ENFORCE_GT(
        length_,
        0UL,
        common::errors::PreconditionNotMet(
            "The memory used in PaddleBuf %d should be greater than 0",
            length_));
    delete[] static_cast<char *>(data_);
    data_ = nullptr;
    length_ = 0;
  }
}

NativeConfig::NativeConfig() {
  LOG(WARNING) << "The paddle::NativeConfig interface is going to be "
                  "deprecated in the next release, please use the latest "
                  "paddle_infer::Config instead.";
}

std::string get_version() {
  std::stringstream ss;
  ss << "version: " << framework::paddle_version() << "\n";
  ss << "commit: " << framework::paddle_commit() << "\n";
  ss << "branch: " << framework::paddle_compile_branch() << "\n";
  return ss.str();
}

void UpdateDllFlag(const char *name, const char *value) {
  std::string ret;
  LOG(WARNING)
      << "The function \"UpdateDllFlag\" is only used to update the flag "
         "on the Windows shared library";
  bool success = paddle::flags::SetFlagValue(name, value);

  PADDLE_ENFORCE_EQ(
      success,
      true,
      common::errors::InvalidArgument(
          "Fail to update flag: %s, please make sure the flag exists.", name));
}

#ifdef PADDLE_WITH_CRYPTO
std::shared_ptr<framework::Cipher> MakeCipher(const std::string &config_file) {
  return framework::CipherFactory::CreateCipher(config_file);
}
#endif

}  // namespace paddle

#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/capi/capi.h"
#endif
