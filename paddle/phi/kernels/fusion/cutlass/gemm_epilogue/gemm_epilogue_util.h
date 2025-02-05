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
#include <cuda_fp16.h>
#include <vector>
#include "paddle/phi/common/bfloat16.h"

#include "paddle/phi/kernels/fusion/cutlass/gemm_epilogue/gemm_epilogue_decl.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

typedef enum {
  MATMUL_ADD,
  MATMUL_ADD_RELU,
  MATMUL_ADD_GELU,
  // MATMUL_ADD_SILU,
  // MATMUL_ADD_LEAKY_RELU,
  // MATMUL_ADD_SIGMOID,
} OpType;

// gemm_epilogue_diff_gpu calculate diff of cutlass output and baseline output,
// you can use them to debug. return value is the max diff between cutlass and
// baseline.
template <typename T>
float gemm_epilogue_diff_gpu(const GemmEpilogueAllParams& params,
                             OpType op_type);

std::string OpType2String(OpType op_type);

int ProfileToGetBestConfig(
    const std::vector<std::function<cutlass::Status(GemmEpilogueAllParams)>>&
        all_func,
    const GemmEpilogueAllParams& params,
    OpType op_type);

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
