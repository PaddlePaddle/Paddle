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
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/common/nvgpu_dev_info.h"

namespace cinn {
namespace common {

std::array<int, 3> NVGPUDevInfo::GetMaxGridDims() const {
  std::array<int, 3> ret;
  ret[0] = prop_.maxGridSize[0];
  ret[1] = prop_.maxGridSize[1];
  ret[2] = prop_.maxGridSize[2];
  return ret;
}

std::array<int, 3> NVGPUDevInfo::GetMaxBlockDims() const {
  std::array<int, 3> ret;
  ret[0] = prop_.maxThreadsDim[0];
  ret[1] = prop_.maxThreadsDim[1];
  ret[2] = prop_.maxThreadsDim[2];
  return ret;
}

int NVGPUDevInfo::GetMultiProcessorCount() const {
  return prop_.multiProcessorCount;
}

int NVGPUDevInfo::GetMaxThreadsPerMultiProcessor() const {
  return prop_.maxThreadsPerMultiProcessor;
}

int NVGPUDevInfo::GetMaxThreadsPerBlock() const {
  return prop_.maxThreadsPerBlock;
}

size_t NVGPUDevInfo::GetMaxSharedMemPerBlock() const {
  return prop_.sharedMemPerBlock;
}

}  // namespace common
}  // namespace cinn
#endif
