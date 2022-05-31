// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle {
namespace framework {

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "paddle/fluid/platform/enforce.h"

inline void debug_gpu_memory_info() {
  int device_num = 0;
  auto err = cudaGetDeviceCount(&device_num);
  PADDLE_ENFORCE_EQ(err, cudaSuccess,
          platform::errors::InvalidArgument("cudaGetDeviceCount failed!"));

  size_t avail{0};
  size_t total{0};
  for (int i = 0; i < device_num; ++i) {
    cudaSetDevice(i);
    auto err = cudaMemGetInfo(&avail, &total);
    PADDLE_ENFORCE_EQ(err, cudaSuccess,
            platform::errors::InvalidArgument("cudaMemGetInfo failed!"));
    VLOG(0) << "update gpu memory!!! "
            << "avail=" << avail << ", "
            << "total=" << total << ", "
            << "ratio=" << (total-avail)/double(total);
  }
}

}; // namespace framework
}; // namespace paddle

