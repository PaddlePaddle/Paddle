/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/kernels/strings/charcases_flag.h"
#include "paddle/phi/kernels/strings/unicode_flag.h"

namespace phi {
namespace strings {

template class UnicodeFlagMap<CPUContext, uint8_t>;
template class UnicodeFlagMap<CPUContext, uint16_t>;

template <>
UnicodeFlagMap<CPUContext, uint8_t>
    UnicodeFlagMap<CPUContext, uint8_t>::m_instance(UNIFLAG_MAP);
template <>
UnicodeFlagMap<CPUContext, uint16_t>
    UnicodeFlagMap<CPUContext, uint16_t>::m_instance(CHARCASES_MAP);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <>
UnicodeFlagMap<GPUContext, uint8_t>::UnicodeFlagMap(uint8_t* flag_map) {
  uint32_t gUMapSize = sizeof(UNIFLAG_MAP);
// Cannot use RecordedGpuMalloc, because it depend on a static instance
// StatRegistry
#ifdef __HIPCC__
  hipMalloc(reinterpret_cast<void**>(&m_charcases_map), gUMapSize);
  phi::backends::gpu::GpuMemcpySync(
      m_charcases_map, flag_map, gUMapSize, hipMemcpyHostToDevice);
#else
  cudaMalloc(reinterpret_cast<void**>(&m_charcases_map), gUMapSize);
  phi::backends::gpu::GpuMemcpySync(
      m_charcases_map, flag_map, gUMapSize, cudaMemcpyHostToDevice);
#endif
}

template <>
UnicodeFlagMap<GPUContext, uint16_t>::UnicodeFlagMap(uint16_t* flag_map) {
  uint32_t gCMapSize = sizeof(CHARCASES_MAP);
#ifdef __HIPCC__
  hipMalloc(reinterpret_cast<void**>(&m_charcases_map), gCMapSize);
  phi::backends::gpu::GpuMemcpySync(
      m_charcases_map, flag_map, gCMapSize, hipMemcpyHostToDevice);
#else
  cudaMalloc(reinterpret_cast<void**>(&m_charcases_map), gCMapSize);
  phi::backends::gpu::GpuMemcpySync(
      m_charcases_map, flag_map, gCMapSize, cudaMemcpyHostToDevice);
#endif
}

template <>
UnicodeFlagMap<GPUContext, uint16_t>::~UnicodeFlagMap() {
#ifdef __HIPCC__
  hipFree(reinterpret_cast<void*>(m_charcases_map));
#else
  cudaFree(reinterpret_cast<void*>(m_charcases_map));
#endif
}

template <>
UnicodeFlagMap<GPUContext, uint8_t>::~UnicodeFlagMap() {
#ifdef __HIPCC__
  hipFree(reinterpret_cast<void*>(m_charcases_map));
#else
  cudaFree(reinterpret_cast<void*>(m_charcases_map));
#endif
}

template class UnicodeFlagMap<GPUContext, uint8_t>;
template class UnicodeFlagMap<GPUContext, uint16_t>;

template <>
UnicodeFlagMap<GPUContext, uint8_t>
    UnicodeFlagMap<GPUContext, uint8_t>::m_instance(UNIFLAG_MAP);
template <>
UnicodeFlagMap<GPUContext, uint16_t>
    UnicodeFlagMap<GPUContext, uint16_t>::m_instance(CHARCASES_MAP);
#endif

}  // namespace strings
}  // namespace phi
