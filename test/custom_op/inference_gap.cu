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

#include "paddle/extension.h"

#define CHECK_GPU_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

template <typename T>
__global__ void globalAvgPool(const T *input,
                              T *output,
                              const int32_t h,
                              const int32_t w) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t len = h * w;
  const T *fmIn = input + idx * len;
  T *fmOut = output + idx;
  T sum = 0;
  int32_t i = 0;
  while (i < len) {
    sum += *fmIn++;
    ++i;
  }
  fmOut[0] = sum / (T)len;
}

void call_kernel(dim3 gridSize,
                 dim3 blockSize,
                 size_t share_M,
                 const cudaStream_t &stream,
                 const float *input,
                 float *output,
                 const int h,
                 const int w) {
  globalAvgPool<<<gridSize, blockSize, share_M, stream>>>(input, output, h, w);
}
