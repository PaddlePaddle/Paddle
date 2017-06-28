/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#ifndef PADDLE_ONLY_CPU

#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

namespace paddle {
namespace platform {

inline void throw_on_error(cudaError_t e, const char* message) {
  if (e) {
    throw thrust::system_error(e, thrust::cuda_category(), message);
  }
}

int GetDeviceCount(void) {
  int count;
  throw_on_error(cudaGetDeviceCount(&count), "cudaGetDeviceCount failed");
  return count;
}

}  // namespace platform
}  // namespace paddle

#endif  // PADDLE_ONLY_CPU
