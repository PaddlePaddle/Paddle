// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_profiler_api.h>

#include <string>

#include "paddle/phi/backends/dynload/nvtx.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace platform {

void CudaProfilerInit(const std::string& output_file,
                      const std::string& output_mode,
                      const std::string& config_file);

void CudaProfilerStart();

void CudaProfilerStop();

#ifndef _WIN32
enum class NvtxRangeColor : uint32_t {
  Black = 0x00000000,
  Red = 0x00ff0000,
  Green = 0x0000ff00,
  Blue = 0x000000ff,
  White = 0x00ffffff,
  Yellow = 0x00ffff00,
};

void CudaNvtxRangePush(const std::string& name,
                       const NvtxRangeColor color = NvtxRangeColor::Green);

void CudaNvtxRangePop();
#endif

}  // namespace platform
}  // namespace paddle
