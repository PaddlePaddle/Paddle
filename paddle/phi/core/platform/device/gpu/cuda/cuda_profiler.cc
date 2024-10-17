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

#include "paddle/phi/core/platform/device/gpu/cuda/cuda_profiler.h"

namespace paddle::platform {

void CudaProfilerInit(const std::string& output_file,
                      const std::string& output_mode,
                      const std::string& config_file) {
#if CUDA_VERSION < 11000
  PADDLE_ENFORCE(output_mode == "kvp" || output_mode == "csv",
                 common::errors::InvalidArgument(
                     "Unsupported cuda profiler output mode, expect `kvp` or "
                     "`csv`, but received `%s`.",
                     output_mode));
  cudaOutputMode_t mode = output_mode == "csv" ? cudaCSV : cudaKeyValuePair;
  PADDLE_ENFORCE_GPU_SUCCESS(
      cudaProfilerInitialize(config_file.c_str(), output_file.c_str(), mode));
#endif
}

void CudaProfilerStart() { PADDLE_ENFORCE_GPU_SUCCESS(cudaProfilerStart()); }

void CudaProfilerStop() { PADDLE_ENFORCE_GPU_SUCCESS(cudaProfilerStop()); }

#ifndef _WIN32
void CudaNvtxRangePush(const std::string& name, const NvtxRangeColor color) {
  nvtxEventAttributes_t eventAttrib;
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(color);
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = name.c_str();

  phi::dynload::nvtxRangePushEx(&eventAttrib);
}

void CudaNvtxRangePop() { phi::dynload::nvtxRangePop(); }
#endif

}  // namespace paddle::platform
