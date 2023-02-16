// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>
#include "paddle/phi/kernels/fusion/cutlass/int4_gemm/int4_gemm_decl.h"

#include "cutlass/cutlass.h"

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace fusion {
namespace cutlass_gemm_internal {
#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

int ProfileToGetBestConfig(
    const std::vector<std::function<cutlass::Status(GemmAllParams)>>
        &gemm_funcs,
    GemmAllParams params);

inline int getSMVersion() {
  const int device = phi::backends::gpu::GetCurrentDeviceId();
  const phi::gpuDeviceProp prop =
      phi::backends::gpu::GetDeviceProperties(device);
  return prop.major * 10 + prop.minor;
}

template <typename Destination, typename Source>
void __global__ DynamicConvert(Source const *s, Destination *t, int N);

template <typename T>
void __global__ ExpendKernel(const T *vector,
                             T *matrix,
                             const int n,
                             const int m,
                             const int col_major = 0);

template <typename T, typename Context>
void ConvertDataToInt4(const Context &ctx,
                       const DenseTensor &source,
                       cutlass::int4b_t *output,
                       const size_t source_size,
                       const bool transpose);

template <typename T>
void ConvertDataToInt4(const T *source,
                       cutlass::int4b_t *output,
                       const size_t source_size);

template <typename Source, typename Target>
void ConvertData(const Source *source,
                 Target *output,
                 const size_t source_size);
}  // namespace cutlass_gemm_internal
}  // namespace fusion
}  // namespace phi
