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

#pragma once

#include <grpc++/grpc++.h>
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace paddle {
namespace operators {
namespace detail {

// A ZeroCopyInputStream that reads from a grpc::ByteBuffer.
class GrpcByteBufferSource
    : public ::google::protobuf::io::ZeroCopyInputStream {
 public:
  GrpcByteBufferSource();
  bool Init(const ::grpc::ByteBuffer& src);  // Can be called multiple times.
  bool Next(const void** data, int* size) override;
  void BackUp(int count) override;
  bool Skip(int count) override;
  ::google::protobuf::int64 ByteCount() const override;

 private:
  std::vector<::grpc::Slice> slices_;
  size_t cur_;       // Current slice index.
  int left_;         // Number of bytes in slices_[cur_] left to yield.
  const char* ptr_;  // Address of next byte in slices_[cur_] to yield.
  ::google::protobuf::int64 byte_count_;
};

}  // namespace detail
}  // namespace operators
}  // namespace paddle
