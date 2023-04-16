/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <stddef.h>

#include "paddle/phi/backends/cpu/cpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

#include "paddle/phi/backends/gpu/gpu_info.h"

namespace phi {

inline size_t Alignment(size_t size,
                        const phi::Place &place,
                        int align_size = -1) {
  size_t alignment = 0;
  if (align_size > 0) {
    alignment = align_size;
  } else {
    alignment = 1024;
    if (place.GetType() == phi::AllocationType::CPU) {
      alignment = phi::backends::cpu::CpuMinChunkSize();
    } else {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      alignment = phi::backends::gpu::GpuMinChunkSize();
#elif defined(PADDLE_WITH_XPU)
      alignment = alignment;
#else
      PADDLE_THROW(phi::errors::PreconditionNotMet(
          "Fluid is not compiled with CUDA/XPU/NPU."));
#endif
    }
  }
  if (place.GetType() == phi::AllocationType::NPU) {
    size += 32;  // required by ascendcl
  }
  size_t remaining = size % alignment;
  return remaining == 0 ? size : size + (alignment - remaining);
}

}  // namespace phi
