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

#include "paddle/fluid/operators/detail/variable_response.h"

namespace paddle {
namespace operators {
namespace detail {

bool VariableResponse::ReadRaw(::google::protobuf::io::CodedInputStream* input,
                               const platform::DeviceContext& dev_ctx,
                               platform::Place place, void* dest, int size) {
  const void* data = NULL;
  int size_to_write = 0;
  int length = size;
  int total_written = 0;

  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    auto& gpu_dev_ctx =
        static_cast<const platform::CUDADeviceContext&>(dev_ctx);
    platform::CPUPlace cpu;

    char* p = reinterpret_cast<char*>(dest);
    while (total_written < length) {
      if (!input->GetDirectBufferPointer(&data, &size_to_write)) {
        return false;
      }
      // NOTE: if raw buffer is large and have two neighbor fields of raw
      // buffers GetDirectBufferPointer can get all of them, use length to
      // truncate it.
      if (total_written + size_to_write > length) {
        size_to_write = length - total_written;
      }
      memory::Copy(boost::get<platform::CUDAPlace>(place),
                   reinterpret_cast<void*>(p), cpu, data, size_to_write,
                   gpu_dev_ctx.stream());
      p += size_to_write;
      total_written += size_to_write;

      input->Skip(size_to_write);
    }
    gpu_dev_ctx.Wait();
#else
    PADDLE_THROW("Unexpected branch");
#endif
    return true;
  }

  char* p = reinterpret_cast<char*>(dest);
  while (total_written < length) {
    if (!input->GetDirectBufferPointer(&data, &size_to_write)) {
      return false;
    }
    // NOTE: if raw buffer is large and have two neighbor fields of raw buffers
    // GetDirectBufferPointer can get all of them, use length to truncate it.
    if (total_written + size_to_write > length) {
      size_to_write = length - total_written;
    }
    // TODO(gongwb): can we avoid copy?
    platform::CPUPlace cpu;
    memory::Copy(cpu, reinterpret_cast<void*>(p), cpu, data, size_to_write);

    p += size_to_write;
    total_written += size_to_write;

    input->Skip(size_to_write);
  }

  return true;
}

};  // namespace detail
};  // namespace operators
};  // namespace paddle
