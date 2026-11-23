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

// scale.cu: the scale kernel declared in scale.h.
// Matches k_scale_range of gin_overlap (256 threads by default): each element
// is __fmul_rn(data[i], scale).
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/scale.h"

namespace paddle {
namespace distributed {
namespace {

__global__ void ScaleRangeKernel(float* data, size_t numel, float scale) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < numel;
       i += stride) {
    data[i] = __fmul_rn(data[i], scale);
  }
}

}  // namespace

cudaError_t LaunchScaleRange(float* data,
                             size_t numel,
                             float scale,
                             int num_blocks,
                             cudaStream_t stream,
                             int num_threads) {
  ScaleRangeKernel<<<num_blocks, num_threads, 0, stream>>>(data, numel, scale);
  return cudaGetLastError();
}

}  // namespace distributed
}  // namespace paddle
