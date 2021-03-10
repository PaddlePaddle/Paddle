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
#include "gflags/gflags.h"
#include "paddle/fluid/framework/commit.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle_infer {

HostBuffer::HostBuffer(HostBuffer &&other)
    : data_(other.data_),
      length_(other.length_),
      memory_owned_(other.memory_owned_) {
  other.memory_owned_ = false;
  other.data_ = nullptr;
  other.length_ = 0;
}

HostBuffer::HostBuffer(const HostBuffer &other) { *this = other; }

HostBuffer &HostBuffer::operator=(const HostBuffer &other) {
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
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "Invalid argument, null pointer data with length %u is passed",
          other.length()));

    length_ = other.length();
    memory_owned_ = true;
  }
  return *this;
}

HostBuffer &HostBuffer::operator=(HostBuffer &&other) {
  // only the buffer with external memory can be copied
  data_ = other.data_;
  length_ = other.length_;
  memory_owned_ = other.memory_owned_;
  other.data_ = nullptr;
  other.length_ = 0;
  other.memory_owned_ = false;
  return *this;
}

void HostBuffer::Resize(size_t length) {
  // Only the owned memory can be reset, the external memory can't be changed.
  if (length_ >= length) return;
  if (memory_owned_) {
    Free();
    data_ = new char[length];
    length_ = length;
    memory_owned_ = true;
  } else {
    PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
        "The memory is allocated externally, can not Resized"));
  }
}

void HostBuffer::Reset(void *data, size_t length) {
  Free();
  memory_owned_ = false;
  data_ = data;
  length_ = length;
}

void HostBuffer::Free() {
  if (memory_owned_ && data_) {
    PADDLE_ENFORCE_GT(
        length_, 0UL,
        paddle::platform::errors::PreconditionNotMet(
            "The memory used in HostBuffer %d should be greater than 0",
            length_));
    delete[] static_cast<char *>(data_);
    data_ = nullptr;
    length_ = 0;
  }
}

}  // namespace paddle_infer

namespace paddle {

size_t PaddleDtypeSize(PaddleDType dtype) {
  switch (dtype) {
    case PaddleDType::FLOAT32:
      return sizeof(float);
    case PaddleDType::INT64:
      return sizeof(int64_t);
    case PaddleDType::INT32:
      return sizeof(int32_t);
    case PaddleDType::UINT8:
      return sizeof(uint8_t);
    default:
      PADDLE_THROW(paddle::platform::errors::InvalidArgument(
          "The dtype must be one in the list: FLOAT32, INT64, INT32, UINT8. "
          "But it is `%d`",
          static_cast<int>(dtype)));
      return 0;
  }
}

NativeConfig::NativeConfig() {
  LOG(WARNING) << "The paddle::NativeConfig interface is going to be "
                  "deprecated in the next release, plase use the latest "
                  "paddle_infer::Config instead.";
}

std::string get_version() {
  std::stringstream ss;
  ss << "version: " << framework::paddle_version() << "\n";
  ss << "commit: " << framework::paddle_commit() << "\n";
  ss << "branch: " << framework::paddle_compile_branch() << "\n";
  return ss.str();
}

std::string UpdateDllFlag(const char *name, const char *value) {
  std::string ret;
  LOG(WARNING)
      << "The function \"UpdateDllFlag\" is only used to update the flag "
         "on the Windows shared library";
  ret = ::GFLAGS_NAMESPACE::SetCommandLineOption(name, value);

  PADDLE_ENFORCE_EQ(
      ret.empty(), false,
      platform::errors::InvalidArgument(
          "Fail to update flag: %s, please make sure the flag exists.", name));
  LOG(INFO) << ret;
  return ret;
}

#ifdef PADDLE_WITH_CRYPTO
std::shared_ptr<framework::Cipher> MakeCipher(const std::string &config_file) {
  return framework::CipherFactory::CreateCipher(config_file);
}
#endif

}  // namespace paddle
