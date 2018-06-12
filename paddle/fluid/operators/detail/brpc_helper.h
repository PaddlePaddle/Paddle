/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include "brpc/channel.h"

namespace paddle {
namespace operators {
namespace detail {

class IOBufParser {
 public:
  IOBufParser(const framework::Scope* scope,
              const platform::DeviceContext* dev_ctx, bool create_scope = false)
      : scope_(scope), dev_ctx_(dev_ctx), create_scope_(create_scope) {
    if (create_scope) {
      local_scope_ = &scope->NewScope();
    }
  }

  virtual ~IOBufParser() {
    if (create_scope_) {
      scope_->DeleteScope(local_scope_);
    }
  }

  bool Parse(butil::IObuf* iobuf) {
    if (!iobuf->Next(&data_, &size)) {
      return false;
    }

    while (1) {
      int tag = -1;
      if (!GetInt(iobuf, &tag)) {
        break;
      }

      int len = 0;
      if (!GetInt(iobuf, &len)) {
        return false;
      }

      if (ReadRaw(tag, iobuf, len)) {
        return false;
      }
    }

    return true;
  }

 private:
  bool ReadRaw(int tag, butil::IOBuf, len) {
    switch (tag) {
      case sendrecv::kRowsFieldNumber: {
        break;
      }

      case sendrecv::kSerializedFieldNumber: {
        break;
      }
      default: {
        PADDLE_ENFORCE(false, "not supported:%d field number", tag);
        return false;
      }
    }

    return true;
  }
  bool GetInt(butil::IOBuf* iobuf, int* tag) {
    char buf[4] = {0};
    int pos = 0;

    while (1) {
      while (offset_ < size) {
        buf[pos++] = data_[offset++];
      }

      if (pos >= 4) {
        *tag = *reinterpret_cast<int*>(&buf);
        return true;
      }

      offset_ = 0;
      if (!iobuf->Next(&data_, &size)) {
        return false;
      }
    }

    return true;
  }

 private:
  const framework::Scope* scope_ = nullptr;
  const platform::DeviceContext* dev_ctx_ = nullptr;
  bool create_scope_ = false;
  framework::Scope* local_scope_ = nullptr;

  // current block
  void* data_ = nullptr;
  int size = 0;
  int offset_ = 0;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
