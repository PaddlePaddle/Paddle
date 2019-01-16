// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "common.h"

void RandomizeFloat(void* dest, const int count, const int seed) {
  float* ptr = (float*)dest;
  curandGenerator_t gen;
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
  CURAND_CHECK(curandGenerateUniform(gen, ptr, count));
  CURAND_CHECK(curandDestroyGenerator(gen));
  CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void KeFeedInputFloat(float* dest, const int count, float* src,
                                 const int size) {
  int offset = (threadIdx.x + blockDim.x * blockIdx.x) % size;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    dest[i] = src[offset];
    offset = (offset + 1) % size;
  }
}

void FeedInputFloat(float* dest, const int count, const float* src,
                    const int size) {
  float* g_src;
  CUDA_CHECK(cudaMalloc((void**)&g_src, size * sizeof(float)));
  CUDA_CHECK(
      cudaMemcpy(g_src, src, size * sizeof(float), cudaMemcpyHostToDevice));
  KeFeedInputFloat<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(dest, count, g_src,
                                                            size);
}
