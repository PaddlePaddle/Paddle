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
#include <map>
#include <vector>

namespace phi {
namespace fusion {
namespace cutlass_internal {

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
  int groups;
  // const phi::GPUContext *ctx;
  cudaStream_t stream;
  float alpha;  // for leaky_relu use
  int sm_version = 75;
  void *workspace = nullptr;
} ConvAllParams;

// Below functions are provided by cutlass, they are called by phi.
extern "C" void Conv2dBiasAddRelu(ConvAllParams params);
extern "C" void Conv2dBiasRelu(ConvAllParams params);
extern "C" void Conv2dBiasLeakyRelu(ConvAllParams params);
extern "C" void Conv2dBiasSilu(ConvAllParams params);
extern "C" void Conv2dBias(ConvAllParams params);
extern "C" void Conv2dBiasSigmoid(ConvAllParams params);

extern "C" void Conv2dDepthwiseBias(ConvAllParams params);
extern "C" void Conv2dDepthwiseBiasRelu(ConvAllParams params);
extern "C" void Conv2dDepthwiseBiasSigmoid(ConvAllParams params);
extern "C" void Conv2dDepthwiseBiasSilu(ConvAllParams params);

extern "C" int HelloFromCutlassConv2d(int a, int b);

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
