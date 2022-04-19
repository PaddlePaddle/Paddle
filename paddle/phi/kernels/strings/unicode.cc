/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/strings/unicode.h"
#include <utf8proc.h>
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/kernels/strings/unicode_flag.h"

namespace phi {
namespace strings {

static const void* utils_map[4] = {nullptr};
static uint16_t CHARCASES_MAP[65536] = {0};

const uint8_t* GetUniFlagMap() {
  if (utils_map[1] == nullptr) {
    utils_map[1] = UNIFLAG_MAP;
  }
  return reinterpret_cast<const uint8_t*>(utils_map[1]);
}

const uint16_t* GetCharcasesMap() {
  if (utils_map[0] == nullptr) {
    for (uint32_t i = 0; i < 65536; ++i) {
      if (utf8proc_islower(i)) {
        CHARCASES_MAP[i] = utf8proc_toupper(i);
      } else if (utf8proc_isupper(i)) {
        CHARCASES_MAP[i] = utf8proc_tolower(i);
      }
    }
    utils_map[0] = CHARCASES_MAP;
  }
  return reinterpret_cast<const uint16_t*>(utils_map[0]);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

const uint8_t* GetGPUUniflagMap() {
  if (utils_map[3] == nullptr) {
    const uint8_t* cpu_uniflag = GetUniFlagMap();
    auto size = sizeof(UNIFLAG_MAP);
    uint8_t* gpu_uniflag;
#ifdef PADDLE_WITH_HIP
    hipMalloc(reinterpret_cast<void**>(&gpu_uniflag), size);
    phi::backends::gpu::GpuMemcpySync(
        gpu_uniflag, cpu_uniflag, size, hipMemcpyHostToDevice);
#else
    cudaMalloc(reinterpret_cast<void**>(&gpu_uniflag), size);
    phi::backends::gpu::GpuMemcpySync(
        gpu_uniflag, cpu_uniflag, size, cudaMemcpyHostToDevice);
#endif
    utils_map[3] = gpu_uniflag;
  }
  return reinterpret_cast<const uint8_t*>(utils_map[3]);
}

const uint16_t* GetGPUCharcasesMap() {
  if (utils_map[2] == nullptr) {
    const uint16_t* cpu_charcases = GetCharcasesMap();
    auto size = sizeof(CHARCASES_MAP);
    uint16_t* gpu_charcases;
#ifdef PADDLE_WITH_HIP
    hipMalloc(reinterpret_cast<void**>(&gpu_charcases), size);
    phi::backends::gpu::GpuMemcpySync(
        gpu_charcases, cpu_charcases, size, hipMemcpyHostToDevice);
#else
    cudaMalloc(reinterpret_cast<void**>(&gpu_charcases), size);
    phi::backends::gpu::GpuMemcpySync(
        gpu_charcases, cpu_charcases, size, cudaMemcpyHostToDevice);
#endif
    utils_map[2] = gpu_charcases;
  }
  return reinterpret_cast<const uint16_t*>(utils_map[2]);
}
#endif

}  // namespace strings
}  // namespace phi
