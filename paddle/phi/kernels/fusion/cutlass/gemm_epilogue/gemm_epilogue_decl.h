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
#include <map>
#include <vector>

namespace phi {
namespace fusion {
namespace cutlass_internal {

typedef enum {
  fp32,
  fp16,
  bf16,
} GemmEpilogueDataType;

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
  GemmEpilogueDataType data_type;
  bool isVec_bias = true;
  int sm_version = 80;
  float leaky_alpha = 1.0;
  void *workspace = nullptr;
} GemmEpilogueAllParams;

// Below functions are provided by cutlass, they are called by phi.
extern "C" void MatmulAdd(GemmEpilogueAllParams params);
extern "C" void MatmulAddRelu(GemmEpilogueAllParams params);
extern "C" void MatmulAddGelu(GemmEpilogueAllParams params);
// extern "C" void MatmulAddLeakyRelu(GemmEpilogueAllParams params);
// extern "C" void MatmulAddSilu(GemmEpilogueAllParams params);
// extern "C" void MatmulAddSigmoid(GemmEpilogueAllParams params);

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
