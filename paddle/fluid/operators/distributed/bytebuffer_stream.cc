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

// NOTE: This file was originally created by tensorflow
//       (https://github.com/tensorflow/tensorflow/) we borrow this
//       file and did some modifications so that we can send gRPC
//       requests without too much copying of the tensor data.

#include "paddle/fluid/operators/distributed/bytebuffer_stream.h"

namespace paddle {
namespace operators {
namespace distributed {

GrpcByteBufferSource::GrpcByteBufferSource() {}

bool GrpcByteBufferSource::Init(const grpc::ByteBuffer& src) {
  cur_ = -1;
  left_ = 0;
  ptr_ = nullptr;
  byte_count_ = 0;
  bool ok = src.Dump(&slices_).ok();
  if (!ok) {
    slices_.clear();
  }
  return ok;
}

bool GrpcByteBufferSource::Next(const void** data, int* size) {
  // Use loop instead of if in case buffer contained empty slices.
  while (left_ == 0) {
    // Advance to next slice.
    cur_++;
    if (cur_ >= slices_.size()) {
      return false;
    }
    const ::grpc::Slice& s = slices_[cur_];
    left_ = s.size();
    ptr_ = reinterpret_cast<const char*>(s.begin());
  }

  *data = ptr_;
  *size = left_;
  byte_count_ += left_;
  ptr_ += left_;
  left_ = 0;
  return true;
}

void GrpcByteBufferSource::BackUp(int count) {
  ptr_ -= count;
  left_ += count;
  byte_count_ -= count;
}

bool GrpcByteBufferSource::Skip(int count) {
  const void* data;
  int size;
  while (Next(&data, &size)) {
    if (size >= count) {
      BackUp(size - count);
      return true;
    }
    // size < count;
    count -= size;
  }
  // error or we have too large count;
  return false;
}

google::protobuf::int64 GrpcByteBufferSource::ByteCount() const {
  return byte_count_;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
