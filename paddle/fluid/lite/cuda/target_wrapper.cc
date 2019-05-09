// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/cuda/target_wrapper.h"

namespace paddle {
namespace lite {

using TargetW = TargetWrapper<TARGET(kCUDA), cudaStream_t, cudaEvent_t>;

void* TargetW::Malloc(size_t size) {
  void* ptr{};
  CHECK_EQ(cudaSuccess, cudaMalloc(&ptr, size));
  return ptr;
}

void TargetW::Free(void* ptr) { CHECK_EQ(cudaSuccess, cudaFree(ptr)); }

void TargetW::MemcpySync(void* dst, const void* src, size_t size,
                         IoDirection dir) {
  switch (dir) {
    case IoDirection::DtoD:
      CHECK(cudaSuccess ==
            cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
      break;
    case IoDirection::HtoD:
      CHECK(cudaSuccess == cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
      break;
    case IoDirection::DtoH:
      CHECK(cudaSuccess == cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

void TargetW::MemcpyAsync(void* dst, const void* src, size_t size,
                          IoDirection dir, const stream_t& stream) {
  switch (dir) {
    case IoDirection::DtoD:
      CHECK(cudaSuccess ==
            cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
      break;
    case IoDirection::HtoD:
      CHECK(cudaSuccess ==
            cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
      break;
    case IoDirection::DtoH:
      CHECK(cudaSuccess ==
            cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

}  // namespace lite
}  // namespace paddle
