#include "hip/hip_runtime.h"
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

#include "hl_base.h"
#include "hl_batch_transpose.h"

const int TILE_DIM = 64;
const int BLOCK_ROWS = 16;

// No bank-conflict transpose for a batch of data.
__global__ void batchTransposeNoBankConflicts(
    real* odata, const real* idata, int numSamples, int width, int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  const int x = hipBlockIdx_x * TILE_DIM + hipThreadIdx_x;
  const int y = hipBlockIdx_y * TILE_DIM + hipThreadIdx_y;
  const int sampleId = hipBlockIdx_z;
  if (sampleId > numSamples) return;
  if (x < width) {
    for (int j = hipThreadIdx_y; j < TILE_DIM && j < height - y + hipThreadIdx_y;
         j += BLOCK_ROWS)
      tile[j][hipThreadIdx_x] =
          idata[sampleId * width * height + (y + j - hipThreadIdx_y) * width + x];
  }

  __syncthreads();

  // The matrix is tranposed. Thus height is new width, and width is new height.
  const int newX = hipBlockIdx_y * TILE_DIM + hipThreadIdx_x;
  const int newY = hipBlockIdx_x * TILE_DIM + hipThreadIdx_y;
  if (newX >= height) {
    return;
  }
  for (int j = hipThreadIdx_y; j < TILE_DIM && j < width - newY + hipThreadIdx_y;
       j += BLOCK_ROWS)
    odata[sampleId * width * height + (newY + j - hipThreadIdx_y) * height +
          newX] = tile[hipThreadIdx_x][j];
}

void batchTranspose(
    const real* input, real* output, int width, int height, int batchSize) {
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
  dim3 dimGrid(DIVUP(width, TILE_DIM), DIVUP(height, TILE_DIM), batchSize);
  hipLaunchKernelGGL((batchTransposeNoBankConflicts), dim3(dimGrid), dim3(dimBlock), 0, STREAM_DEFAULT, 
      output, input, batchSize, width, height);

  CHECK_SYNC("batchTranspose failed!");
}
