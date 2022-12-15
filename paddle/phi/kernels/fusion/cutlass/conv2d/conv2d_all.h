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
#include <glog/logging.h>
#include <iostream>
#include <map>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace fusion {

#define CUTLASS_CHECK(status)                                                \
  if (status != cutlass::Status::kSuccess) {                                 \
    VLOG(3)                                                                  \
        << "Cutlass can not deal with this problem size, skip this kernel!"; \
    return status;                                                           \
  }

#define WARMUP 10
#define REPEAT 100

typedef struct {
  const half *input;
  const half *weight;
  const half *bias;
  const half *residual;
  half *output;
  int batch;
  int ic;
  int ih;
  int iw;
  int kh;
  int kw;
  int oc;
  int pad_h0;
  int pad_h1;
  int pad_w0;
  int pad_w1;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int oh;
  int ow;
  cudaStream_t stream;
  float alpha;  // for leaky_relu use
} ConvAllParams;

// Below functions are provided by cutlass, these are called by phi.
void cutlass_conv2d_bias_add_relu(ConvAllParams params);
void cutlass_conv2d_bias_relu_few_channels(ConvAllParams params);
void cutlass_conv2d_bias_relu(ConvAllParams params);
void cutlass_conv2d_bias_leaky_relu(ConvAllParams params);
void cutlass_conv2d_bias_silu(ConvAllParams params);
void cutlass_conv2d_bias(ConvAllParams params);

}  // namespace fusion
}  // namespace phi
