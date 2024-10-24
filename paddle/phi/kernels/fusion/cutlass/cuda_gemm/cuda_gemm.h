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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace phi {
namespace fusion {
namespace cutlass_internal {

typedef struct {
  void const* act;
  void const* weight;
  void* output;
  int32_t m, n, k;
  int inputType;
  int outputType;
  cudaStream_t stream;
} GemmParams;

// Below functions are provided, they are called by phi.
extern "C" bool cudaGemmDispatcher(GemmParams params);

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
