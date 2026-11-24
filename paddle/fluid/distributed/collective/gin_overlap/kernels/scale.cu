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

// scale.cu: multiplies a contiguous range of floats of the send window by
// scale, element by element (__fmul_rn). Launched on the compute stream by
// GradCommEngine::scale_range.
#include "paddle/fluid/distributed/collective/gin_overlap/kernels/kernels.h"

namespace gin_overlap {
namespace {

__global__ void k_scale_range(float* data, size_t numel, float scale) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < numel;
       i += stride)
    data[i] = __fmul_rn(data[i], scale);
}

}  // namespace

cudaError_t launch_scale_range(float* data,
                               size_t numel,
                               float scale,
                               int num_blocks,
                               cudaStream_t stream) {
  k_scale_range<<<num_blocks, 256, 0, stream>>>(data, numel, scale);
  return cudaGetLastError();
}

cudaError_t preload_scale_kernels() {
  cudaFuncAttributes fa;
  const cudaError_t err = cudaFuncGetAttributes(&fa, k_scale_range);
  return err;
}

}  // namespace gin_overlap
