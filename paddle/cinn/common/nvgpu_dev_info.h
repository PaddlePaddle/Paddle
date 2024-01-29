// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#pragma once

#ifdef CINN_WITH_CUDA

#include <ostream>
#include <string>
#include <vector>

#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/dev_info_base.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/common/target.h"

namespace cinn {
namespace common {

class NVGPUDevInfo : public DevInfoBase {
 public:
  // Note(LiuYang):Since CI own zero GPU while compiling CINN_WITH_CUDA, here
  // I can't use CUDA_CALL to check ret of cuda_runtime_api, devops should
  // always check IsValid before use other api functions.
  explicit NVGPUDevInfo(int device_num = 0) : DevInfoBase(device_num) {
    if (cudaGetDeviceProperties(&prop_, device_num) != cudaSuccess)
      is_valid_ = false;
  }

  bool IsValid() const { return is_valid_; }
  std::array<int, 3> GetMaxGridDims() const;
  std::array<int, 3> GetMaxBlockDims() const;
  int GetMultiProcessorCount() const;
  int GetMaxThreadsPerMultiProcessor() const;
  int GetMaxThreadsPerBlock() const;
  size_t GetMaxSharedMemPerBlock() const;

 private:
  bool is_valid_{true};
  cudaDeviceProp prop_;
};
}  // namespace common
}  // namespace cinn
#endif
