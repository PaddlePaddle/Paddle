// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_decl.h"

#include "glog/logging.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {
#define CUTLASS_CHECK(status)                                                \
  if (status != cutlass::Status::kSuccess) {                                 \
    VLOG(3)                                                                  \
        << "Cutlass can not deal with this problem size, skip this kernel!"; \
    return status;                                                           \
  }

typedef enum {
  CONV2D_BIAS,
  CONV2D_BIAS_RELU,
  CONV2D_BIAS_ADD_RELU,
  CONV2D_BIAS_SILU,
  CONV2D_BIAS_LEAKY_RELU,
  CONV2D_BIAS_SIGMOID,
  CONV2D_BIAS_SILU_ADD,
  CONV2D_DEPTHWISE_BIAS,
  CONV2D_DEPTHWISE_BIAS_RELU,
  CONV2D_DEPTHWISE_BIAS_SIGMOID,
  CONV2D_DEPTHWISE_BIAS_SILU,
} OpType;

// conv2d_diff_gpu calculate diff of cutlass output and baseline output, you can
// use them to debug. return value is the max diff between cutlass and baseline.
float conv2d_diff_gpu(const ConvAllParams& params, OpType op_type);

int ProfileToGetBestConfig(
    const std::vector<std::function<cutlass::Status(ConvAllParams)>>& all_func,
    const ConvAllParams& params,
    OpType op_type);

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
