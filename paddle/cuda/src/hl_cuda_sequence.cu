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
#include "hl_device_functions.cuh"
#include "paddle/utils/Logging.h"

__global__ void KeMaxSequenceForward(real *input,
                                     const int *sequence,
                                     real* output,
                                     int *index,
                                     int numSequences,
                                     int dim) {
  int dimIdx = threadIdx.x;
  int sequenceId = blockIdx.x;
  if (sequenceId >= numSequences) return;
  int start = sequence[sequenceId];
  int end = sequence[sequenceId+1];

  for (int i = dimIdx; i < dim; i += blockDim.x) {
    real tmp = -HL_FLOAT_MAX;
    int tmpId = -1;
    for (int insId = start; insId < end; insId++) {
      if (tmp < input[insId*dim + i]) {
        tmp = input[insId*dim + i];
        tmpId = insId;
      }
    }
    output[sequenceId*dim + i] = tmp;
    index[sequenceId*dim + i] = tmpId;
  }
}

void hl_max_sequence_forward(real* input,
                             const int* sequence,
                             real* output,
                             int *index,
                             int numSequences,
                             int dim) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(sequence);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(index);

  dim3 threads(256, 1);
  dim3 grid(numSequences, 1);
  KeMaxSequenceForward<<< grid, threads, 0, STREAM_DEFAULT >>>
      (input, sequence, output, index, numSequences, dim);
  CHECK_SYNC("hl_max_sequence_forward failed");
}

__global__ void KeMaxSequenceBackward(real *outputGrad,
                                      int *index,
                                      real* inputGrad,
                                      int numSequences,
                                      int dim) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int colIdx = idx % dim;
  if (idx < numSequences*dim) {
    int insId = index[idx];
    inputGrad[insId * dim + colIdx] += outputGrad[idx];
  }
}

void hl_max_sequence_backward(real* outputGrad,
                              int *index,
                              real* inputGrad,
                              int numSequences,
                              int dim) {
  CHECK_NOTNULL(outputGrad);
  CHECK_NOTNULL(index);
  CHECK_NOTNULL(inputGrad);

  unsigned int blocks = (numSequences * dim + 128 - 1) / 128;
  dim3 threads(128, 1);
  dim3 grid(blocks, 1);
  KeMaxSequenceBackward<<< grid, threads, 0, STREAM_DEFAULT >>>
      (outputGrad, index, inputGrad, numSequences, dim);
  CHECK_SYNC("hl_max_sequence_backward failed");
}

template<int blockDimX, int blockDimY, int gridDimX, bool AddRow>
__global__ void KeMatrixAddRows(real* output,
                                real* table,
                                int* ids,
                                int numSamples,
                                int tableSize,
                                int dim) {
  int idx = threadIdx.x;
  int idy = threadIdx.y;
  int sampleId = blockIdx.x + idy * gridDimX;

  while (sampleId < numSamples) {
    int tableId = ids[sampleId];
    if ((0 <= tableId) && (tableId < tableSize)) {
      real *outputData = output + sampleId * dim;
      real *tableData = table + tableId * dim;
      for (int i = idx; i < dim; i += blockDimX) {
        if (AddRow == 0) {
          outputData[i] += tableData[i];
        } else {
          paddle::paddleAtomicAdd(&tableData[i], outputData[i]);
        }
      }
    }
    sampleId += blockDimY*gridDimX;
  }
}

template<int blockDimX, int blockDimY, int gridDimX, bool seq2batch, bool isAdd>
__global__
void KeSequence2Batch(real *batch,
                      real *sequence,
                      const int *batchIndex,
                      int seqWidth,
                      int batchCount) {
  int idx = threadIdx.x;
  int idy = threadIdx.y;
  int id = blockIdx.x + idy * gridDimX;
  while (id < batchCount) {
    int seqId = batchIndex[id];
    real* batchData = batch + id*seqWidth;
    real* seqData = sequence + seqId*seqWidth;
    for (int i = idx; i < seqWidth; i += blockDimX) {
      if (seq2batch) {
        if (isAdd) {
          batchData[i] += seqData[i];
        } else {
          batchData[i] = seqData[i];
        }
      } else {
        if (isAdd) {
          seqData[i] += batchData[i];
        } else {
          seqData[i] = batchData[i];
        }
      }
    }
    id += blockDimY*gridDimX;
  }
}

void hl_sequence2batch_copy(real *batch,
                            real *sequence,
                            const int *batchIndex,
                            int seqWidth,
                            int batchCount,
                            bool seq2batch) {
  CHECK_NOTNULL(sequence);
  CHECK_NOTNULL(batch);
  CHECK_NOTNULL(batchIndex);

  dim3 threads(128, 8);
  dim3 grid(8, 1);
  if (seq2batch) {
    KeSequence2Batch<128, 8, 8, 1, 0><<< grid, threads, 0, STREAM_DEFAULT >>>
      (batch, sequence, batchIndex, seqWidth, batchCount);
  } else {
    KeSequence2Batch<128, 8, 8, 0, 0><<< grid, threads, 0, STREAM_DEFAULT >>>
      (batch, sequence, batchIndex, seqWidth, batchCount);
  }
  CHECK_SYNC("hl_sequence2batch_copy failed");
}

void hl_sequence2batch_add(real *batch,
                           real *sequence,
                           int *batchIndex,
                           int seqWidth,
                           int batchCount,
                           bool seq2batch) {
  CHECK_NOTNULL(sequence);
  CHECK_NOTNULL(batch);
  CHECK_NOTNULL(batchIndex);

  dim3 threads(128, 8);
  dim3 grid(8, 1);
  if (seq2batch) {
    KeSequence2Batch<128, 8, 8, 1, 1><<< grid, threads, 0, STREAM_DEFAULT >>>
      (batch, sequence, batchIndex, seqWidth, batchCount);
  } else {
    KeSequence2Batch<128, 8, 8, 0, 1><<< grid, threads, 0, STREAM_DEFAULT >>>
      (batch, sequence, batchIndex, seqWidth, batchCount);
  }
  CHECK_SYNC("hl_sequence2batch_add failed");
}

template<bool normByTimes, bool seq2batch>
__global__
void KeSequence2BatchPadding(real* batch,
                             real* sequence,
                             const int* sequenceStartPositions,
                             const size_t sequenceWidth,
                             const size_t maxSequenceLength,
                             const size_t numSequences) {
  int batchIdx = blockIdx.y;
  int sequenceStart = sequenceStartPositions[batchIdx];
  int sequenceLength = sequenceStartPositions[batchIdx + 1] - sequenceStart;

  int sequenceIdx = blockIdx.x * blockDim.y + threadIdx.y;
  int batchBaseIdx = (sequenceIdx * numSequences + batchIdx) * sequenceWidth;
  int sequenceBaseIdx = (sequenceStart + sequenceIdx) * sequenceWidth;

  real scale = normByTimes ? (1.0f / (real)sequenceLength) : 1.0f;

  if (sequenceIdx < sequenceLength) {
    if (seq2batch) {
      /* sequence -> batch */
      for (int i = threadIdx.x; i < sequenceWidth; i += blockDim.x) {
        batch[batchBaseIdx + i] = scale * sequence[sequenceBaseIdx + i];
      }
    } else {
      /* batch -> sequence */
      for (int i = threadIdx.x; i < sequenceWidth; i += blockDim.x) {
        sequence[sequenceBaseIdx + i] = scale * batch[batchBaseIdx + i];
      }
    }
  } else if (sequenceIdx < maxSequenceLength) {
    if (seq2batch) {
      /* sequence -> batch */
      for (int i = threadIdx.x; i < sequenceWidth; i += blockDim.x) {
        batch[batchBaseIdx + i] = 0;
      }
    }
  }
}

void hl_sequence2batch_copy_padding(real* batch,
                                    real* sequence,
                                    const int* sequenceStartPositions,
                                    const size_t sequenceWidth,
                                    const size_t maxSequenceLength,
                                    const size_t numSequences,
                                    bool normByTimes,
                                    bool seq2batch) {
  CHECK_NOTNULL(batch);
  CHECK_NOTNULL(sequence);
  CHECK_NOTNULL(sequenceStartPositions);

  if (!normByTimes && numSequences == 1) {
    size_t elementCount = maxSequenceLength * sequenceWidth;
    if (seq2batch) {
      /* sequence -> batch */
      hl_memcpy_device2device(batch, sequence, sizeof(real) * elementCount);
    } else {
      /* batch -> sequence */
      hl_memcpy_device2device(sequence, batch, sizeof(real) * elementCount);
    }
    return;
  }

  const int CUDA_BLOCK_SIZE = 512;

  /* At least use 32 threads to copy sequenceWidth elements,
     and at least 8 elements for each thread. */
  int blockDimX = ((((sequenceWidth + 7) >> 3) + 31) >> 5) << 5;
  blockDimX = (blockDimX < CUDA_BLOCK_SIZE) ? blockDimX : CUDA_BLOCK_SIZE;

  int blockDimY = CUDA_BLOCK_SIZE / blockDimX;
  dim3 threads(blockDimX, blockDimY);

  int gridDimX = (maxSequenceLength * blockDimX + CUDA_BLOCK_SIZE - 1) /
      CUDA_BLOCK_SIZE;
  int gridDimY = numSequences;
  dim3 grid(gridDimX, gridDimY);

  if (seq2batch) {
    /* sequence -> batch */
    if (normByTimes) {
      KeSequence2BatchPadding<1, 1><<< grid, threads, 0, STREAM_DEFAULT >>>(
              batch, sequence, sequenceStartPositions,
              sequenceWidth, maxSequenceLength, numSequences);
    } else {
      KeSequence2BatchPadding<0, 1><<< grid, threads, 0, STREAM_DEFAULT >>>(
              batch, sequence, sequenceStartPositions,
              sequenceWidth, maxSequenceLength, numSequences);
    }
  } else {
    /* batch -> sequence */
    if (normByTimes) {
      KeSequence2BatchPadding<1, 0><<< grid, threads, 0, STREAM_DEFAULT >>>(
              batch, sequence, sequenceStartPositions,
              sequenceWidth, maxSequenceLength, numSequences);
    } else {
      KeSequence2BatchPadding<0, 0><<< grid, threads, 0, STREAM_DEFAULT >>>(
              batch, sequence, sequenceStartPositions,
              sequenceWidth, maxSequenceLength, numSequences);
    }
  }

  CHECK_SYNC("hl_sequence2batch_copy_padding failed");
}

__device__ inline float my_rsqrt(float x) {
  return rsqrtf(x);
}

__device__ inline double my_rsqrt(double x) {
  return rsqrt(x);
}

__global__ void KeSequenceAvgForward(real* dst,
                                     real* src,
                                     const int* starts,
                                     int height,
                                     int width,
                                     const int mode) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = gid / width;
  int col = gid % width;

  if (gid < height * width) {
    int start = starts[row];
    int end = starts[row + 1];
    int seqLength = end - start;
    if (seqLength == 0) return;
    real sum = 0.0;
    for (int i = start; i < end; i++) {
      sum += src[i * width + col];
    }
    sum = mode == 1 ? sum :
        (mode == 0 ? sum / seqLength : sum * my_rsqrt((real)seqLength));
    dst[gid] = sum;
  }
}

void hl_sequence_avg_forward(real* dst,
                             real* src,
                             const int* starts,
                             int height,
                             int width,
                             const int mode) {
  CHECK_NOTNULL(dst);
  CHECK_NOTNULL(src);
  CHECK_NOTNULL(starts);

  int block = 512;
  int grid = DIVUP(width * height, 512);

  CHECK(mode == 0 || mode == 1 || mode == 2)
    << "mode error in hl_sequence_avg_forward!";

  KeSequenceAvgForward<<< grid, block, 0, STREAM_DEFAULT >>>
           (dst, src, starts, height, width, mode);
  CHECK_SYNC("hl_sequence_avg_forward failed");
}

__global__ void KeSequenceAvgBackward(real* dst,
                                      real* src,
                                      const int* starts,
                                      int height,
                                      int width,
                                      const int mode) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = gid / width;
  int col = gid % width;

  if (gid < height * width) {
    int start = starts[row];
    int end = starts[row + 1];
    int seqLength = end - start;
    if (seqLength == 0) return;
    real grad = src[gid];
    grad = mode == 1 ? grad :
        (mode == 0 ? grad / seqLength : grad * my_rsqrt((real)seqLength));
    for (int i = start; i < end; i++) {
      dst[i * width + col] += grad;
    }
  }
}

void hl_sequence_avg_backward(real* dst,
                              real* src,
                              const int* starts,
                              int height,
                              int width,
                              const int mode) {
  CHECK_NOTNULL(dst);
  CHECK_NOTNULL(src);
  CHECK_NOTNULL(starts);

  int block = 512;
  int grid = DIVUP(width * height, 512);

  CHECK(mode == 0 || mode == 1 || mode == 2)
    << "mode error in hl_sequence_avg_backward!";

  KeSequenceAvgBackward<<< grid, block, 0, STREAM_DEFAULT >>>
           (dst, src, starts, height, width, mode);
  CHECK_SYNC("hl_sequence_avg_backward failed");
}
