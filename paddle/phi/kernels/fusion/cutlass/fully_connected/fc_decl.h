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
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include <map>
#include <vector>

// namespace phi {
// namespace fusion {
// namespace cutlass_internal {

typedef enum {
  fp32,
  fp16,
  bf16,
} FcDataType;

typedef struct {
  const void *input;
  const void *weight;
  const void *bias;
  void *output;
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldd;
  cudaStream_t stream;
  FcDataType data_type;
  int sm_version = 75;
  float leaky_alpha = 1.0;
} FcAllParams;

// Below functions are provided by cutlass, they are called by phi.
extern "C" void FcBiasRelu(FcAllParams params);
extern "C" void FcBiasLeakyRelu(FcAllParams params);
extern "C" void FcBiasSilu(FcAllParams params);
extern "C" void FcBias(FcAllParams params);
extern "C" void FcBiasSigmoid(FcAllParams params);

// }  // namespace cutlass_internal
// }  // namespace fusion
// }  // namespace phi