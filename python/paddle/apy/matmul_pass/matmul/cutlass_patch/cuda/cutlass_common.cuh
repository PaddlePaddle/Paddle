// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// Definitions shared by all the cutlass based backends (matmul, conv2d, ...).

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <iostream>

#include "cutlass/arch/mma.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"

#include "cutlass_patch/batched_matrix_coord.h"

#define CHECK_CUTLASS(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

namespace ap {

using bfloat16 = nv_bfloat16;

template <typename T, int N>
using Array = cutlass::Array<T, N>;

using MatrixCoord = cutlass_patch::BatchedMatrixCoord;

// Convert CUDA data type to cutlass data type
template <typename T>
struct CutlassDataType {
  using Type = T;
};

template <>
struct CutlassDataType<half> {
  using Type = cutlass::half_t;
};

template <>
struct CutlassDataType<__nv_bfloat16> {
  using Type = cutlass::bfloat16_t;
};

// Math operation performed by the mainloop
template <typename ElementT>
struct GemmOperation {
  using Type = cutlass::arch::OpMultiplyAdd;
};

template <>
struct GemmOperation<float> {
  using Type = cutlass::arch::OpMultiplyAddFastF32;
};

static void *GetWorkspace(size_t workspace_size) {
  static cutlass::device_memory::allocation<uint8_t> workspace;
  if (workspace.size() < workspace_size) {
    workspace.reset(workspace_size);
  }
  return workspace.get();
}

}  // namespace ap
