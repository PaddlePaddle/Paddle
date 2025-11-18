// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/platform/device/xpu/xpu_profiler.h"

namespace paddle::platform {

void CudaProfilerInit(const std::string& output_file,
                      const std::string& output_mode,
                      const std::string& config_file) {}

void CudaProfilerStart() { PADDLE_ENFORCE_XRE_SUCCESS(cudaProfilerStart()); }

void CudaProfilerStop() { PADDLE_ENFORCE_XRE_SUCCESS(cudaProfilerStop()); }

// #ifndef _WIN32
void CudaNvtxRangePush(const std::string& name, const NvtxRangeColor color) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(color);
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = name.c_str();

  phi::dynload::nvtxRangePushEx(&eventAttrib);
}

void CudaNvtxRangePop() { phi::dynload::nvtxRangePop(); }

}  // namespace paddle::platform
