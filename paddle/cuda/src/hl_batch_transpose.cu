/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "hl_batch_transpose.h"
#include "hl_base.h"

const int TILE_DIM = 64;
const int BLOCK_ROWS = 16;

// No bank-conflict transpose for a batch of data.
__global__ void batchTransposeNoBankConflicts(real* odata,
                                              const real* idata,
                                              int numSamples, int width,
                                              int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  const int x = blockIdx.x * TILE_DIM + threadIdx.x;
  const int y = blockIdx.y * TILE_DIM + threadIdx.y;
  const int sampleId = blockIdx.z;
  if (sampleId > numSamples) return;
  if (x < width) {
    for (int j = threadIdx.y; j < TILE_DIM && j < height - y + threadIdx.y;
         j += BLOCK_ROWS)
      tile[j][threadIdx.x] =
          idata[sampleId * width * height + (y + j - threadIdx.y) * width + x];
  }

  __syncthreads();

  // The matrix is tranposed. Thus height is new width, and width is new height.
  const int newX = blockIdx.y * TILE_DIM + threadIdx.x;
  const int newY = blockIdx.x * TILE_DIM + threadIdx.y;
  if (newX >= height) {
    return;
  }
  for (int j = threadIdx.y; j < TILE_DIM && j < width - newY + threadIdx.y;
       j += BLOCK_ROWS)
    odata[sampleId * width * height + (newY + j - threadIdx.y) * height +
          newX] = tile[threadIdx.x][j];
}

void batchTranspose(const real* input, real* output, int width, int height,
                    int batchSize) {
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
  dim3 dimGrid(DIVUP(width, TILE_DIM), DIVUP(height, TILE_DIM), batchSize);
  batchTransposeNoBankConflicts<<<dimGrid, dimBlock, 0, STREAM_DEFAULT>>>
      (output, input, batchSize, width, height);

  CHECK_SYNC("batchTranspose failed!");
}
