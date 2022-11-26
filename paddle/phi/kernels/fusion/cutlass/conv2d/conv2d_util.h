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
#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_all.h"

namespace phi {
namespace fusion {

typedef enum {
  CONV2D_BIAS,
  CONV2D_BIAS_RELU,
  CONV2D_BIAS_ADD_RELU,
  CONV2D_BIAS_SILU
} OpType;

// This two functions calculate diff of cutlass output and baseline output
// We recommend use conv2d_diff_gpu bacause gpu is more fast than cpu
// return value is the max diff between cutlass and baseline
float conv2d_diff_cpu(COMMON_CONV_PARAMS, const half* residual, OpType op_type);
float conv2d_diff_gpu(COMMON_CONV_PARAMS, const half* residual, OpType op_type);

}  // namespace fusion
}  // namespace phi
