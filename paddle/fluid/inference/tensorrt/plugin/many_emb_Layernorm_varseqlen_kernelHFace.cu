// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
// SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/common.cuh"
#include "common/plugin.h"
#include "common/serialize.h"
#include "paddle/fluid/inference/tensorrt/plugin/many_emb_layernorm_varseqlen_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

template <typename T, unsigned TPB>
__global__ void embLayerNormKernelHFace(int32_t ld,
                                        int32_t** inputIds,
                                        int32_t const nbLookupTables,
                                        float const* beta,
                                        float const* gamma,
                                        T** mIdsEmbDev,
                                        int32_t* IdsSize,
                                        T* output) {
  cub::Sum pairSum;
  int32_t const s = blockIdx.x;
  int32_t const b = blockIdx.y;
  int32_t* cuSeqlens = inputIds[0];
  int32_t const sumS = cuSeqlens[b];
  int32_t const s_b = cuSeqlens[b + 1] - sumS;
  if (s >= s_b) {
    return;  // This CTA has nothing to do
  }
  T const rld = T(1.f) / T(ld);
  int32_t const seqPos = sumS + s;
  extern __shared__ int32_t word_id[];

  if (threadIdx.x == 0) {
    for (int i = 1; i < nbLookupTables; ++i) {
      if (static_cast<int32_t const*>(inputIds[i])[seqPos] < 0 ||
          static_cast<int32_t const*>(inputIds[i])[seqPos] >= IdsSize[i]) {
        printf(
            "Error!!!!!!(embLayerNormVarSeqlenPlugin): ID cannot be lookup "
            "table: ID < 0 or ID > max ");
        return;
      } else {
        word_id[i - 1] = static_cast<int32_t const*>(inputIds[i])[seqPos];
      }
    }
  }
  __syncthreads();

  // 2. load pos/tok/word embeddings and add them toghether
  // offset into embeddings is given by wordId * hidden_size
  int32_t const poffset = blockIdx.x * ld;
  int32_t const outOffset = seqPos * ld;
  // the output offset is given by b * (S*hidden_size) + s * hidden_size
  kvp<T> threadData(0, 0);

  for (int32_t it = threadIdx.x; it < ld; it += TPB) {
    T p(mIdsEmbDev[0][poffset + it]);  // pos id
    T val = p;
    for (int i = 1; i < nbLookupTables; ++i) {
      int32_t const offset = word_id[i - 1] * ld;
      val += mIdsEmbDev[i][offset + it];
    }
    output[outOffset + it] = val;

    T const rldval = rld * val;
    threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
  }

  // 3. layer norm on the sum
  layerNorm<T, T, float, TPB>(threadData, ld, outOffset, beta, gamma, output);
}

template <typename T>
int32_t embSkipLayerNormHFace(cudaStream_t stream,
                              int32_t ld,
                              int32_t B,
                              int32_t S,
                              int32_t** inputIds,
                              int32_t const nbLookupTables,
                              float const* beta,
                              float const* gamma,
                              T** mIdsEmbDev,
                              int32_t* IdsSize,
                              T* output) {
  constexpr int32_t tpb = 256;
  dim3 const grid(S, B, 1);
  dim3 const block(tpb, 1, 1);
  size_t cache_size = sizeof(int32_t) * (nbLookupTables - 1);
  embLayerNormKernelHFace<T, tpb><<<grid, block, cache_size, stream>>>(
      ld, inputIds, nbLookupTables, beta, gamma, mIdsEmbDev, IdsSize, output);
  return cudaPeekAtLastError();
}

template int32_t embSkipLayerNormHFace<float>(cudaStream_t,
                                              int32_t,
                                              int32_t,
                                              int32_t,
                                              int32_t**,
                                              int32_t const,
                                              float const*,
                                              float const*,
                                              float**,
                                              int32_t*,
                                              float*);

template int32_t embSkipLayerNormHFace<half>(cudaStream_t,
                                             int32_t,
                                             int32_t,
                                             int32_t,
                                             int32_t**,
                                             int32_t const,
                                             float const*,
                                             float const*,
                                             half**,
                                             int32_t*,
                                             half*);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
