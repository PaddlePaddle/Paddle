/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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

template <bool padding>
__global__ void KeContextProjectionForward(real* input,
                                           const int* sequence,
                                           real* weightData,
                                           real* output,
                                           int inputDim,
                                           int contextLength,
                                           int contextStart,
                                           int beginPad) {
  int idx = threadIdx.x;
  int blockSize = blockDim.x;
  int sequenceId = blockIdx.x;
  int seqStart = sequence[sequenceId];
  int seqEnd = sequence[sequenceId+1];
  real value = 0;

  int instances = seqEnd - seqStart + contextLength - 1;
  output += seqStart * inputDim * contextLength;
  input += seqStart * inputDim;
  for (int k = 0; k <= inputDim / blockSize; k++) {
    if (idx < inputDim) {
      for (int i = 0; i < instances; i++) {
        // i + contextStart;
        if ((i + contextStart) < 0) {
          if (padding) {
            value = weightData[i * inputDim + idx];
          } else {
            continue;
          }
        } else if ((i + contextStart) >= (seqEnd - seqStart)) {
          if (padding) {
            value =
              weightData[(beginPad + i + contextStart - (seqEnd - seqStart)) *
                         inputDim + idx];
          } else {
            continue;
          }
        } else {
          value = input[(i + contextStart) * inputDim + idx];
        }

        int outx = (i - contextLength) < 0 ? i : (contextLength - 1);
        int outy = (i - contextLength) < 0 ? 0 : (i - (contextLength - 1));
        real* output_r =
          output + outy * inputDim * contextLength + outx * inputDim;
        for (int j = outy; j < seqEnd - seqStart; j++) {
          output_r[idx] += value;
          if (j - outy == outx) break;
          output_r += (contextLength - 1) * inputDim;
        }
      }
    }
    idx += blockSize;
  }
}

void hl_context_projection_forward(real* input,
                                   const int* sequence,
                                   real* weightData,
                                   real* output,
                                   int numSequences,
                                   int inputDim,
                                   int contextLength,
                                   int contextStart,
                                   int beginPad,
                                   bool isPadding) {
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(sequence);
  CHECK_NOTNULL(output);
  CHECK(!isPadding || weightData);

  int blockSize = 128;
  int blocksX = numSequences;
  int blocksY = 1;
  dim3 threads(blockSize, 1);
  dim3 grid(blocksX, blocksY);

  if (isPadding) {
    KeContextProjectionForward<true><<< grid, threads, 0, STREAM_DEFAULT >>>
      (input, sequence, weightData, output, inputDim,
       contextLength, contextStart, beginPad);
  } else  {
    KeContextProjectionForward<false><<< grid, threads, 0, STREAM_DEFAULT >>>
      (input, sequence, weightData, output, inputDim,
       contextLength, contextStart, beginPad);
  }
  CHECK_SYNC("hl_context_projection_forward failed");
}

__global__ void KeContextProjectionBackwardData(real* outputGrad,
                                                const int* sequence,
                                                real* inputGrad,
                                                int inputDim,
                                                int contextLength,
                                                int contextStart) {
  int idx = threadIdx.x;
  int blockSize = blockDim.x;
  int sequenceId = blockIdx.x;
  int seqStart = sequence[sequenceId];
  int seqEnd = sequence[sequenceId+1];
  real value = 0;

  int instances = seqEnd - seqStart + contextLength - 1;
  outputGrad += seqStart * inputDim * contextLength;
  inputGrad += seqStart * inputDim;
  for (int k = 0; k <= inputDim / blockSize; k++) {
    if (idx < inputDim) {
      for (int i = 0; i < instances; i++) {
        if ((i + contextStart) < 0) {
          continue;
        } else if ((i + contextStart) >= (seqEnd - seqStart)) {
          continue;
        } else {
          // value = 0;
          value = inputGrad[(i + contextStart) * inputDim + idx];
        }

        int outx = (i - contextLength) < 0 ? i : (contextLength - 1);
        int outy = (i - contextLength) < 0 ? 0 : (i - (contextLength - 1));
        real* output_r =
          outputGrad + outy * inputDim * contextLength + outx * inputDim;
        for (int j = outy; j < seqEnd - seqStart; j++) {
          value += output_r[idx];
          if (j - outy == outx) break;
          output_r += (contextLength - 1) * inputDim;
        }
        inputGrad[(i + contextStart) * inputDim + idx] = value;
      }
    }
    idx += blockSize;
  }
}

void hl_context_projection_backward_data(real* outputGrad,
                                         const int* sequence,
                                         real* inputGrad,
                                         int numSequences,
                                         int inputDim,
                                         int contextLength,
                                         int contextStart) {
  CHECK_NOTNULL(outputGrad);
  CHECK_NOTNULL(sequence);
  CHECK_NOTNULL(inputGrad);

  int blockSize = 128;
  int blocksX = numSequences;
  int blocksY = 1;
  dim3 threads(blockSize, 1);
  dim3 grid(blocksX, blocksY);
  KeContextProjectionBackwardData<<< grid, threads, 0, STREAM_DEFAULT >>>
    (outputGrad, sequence, inputGrad, inputDim, contextLength, contextStart);
  CHECK_SYNC("hl_context_projection_backward_data failed");
}

template<int THREADS_X, int THREADS_Y>
__global__ void KeContextProjectionBackwardWeight(real* outputGrad,
                                                  const int* sequence,
                                                  real* weightGrad,
                                                  int numSequences,
                                                  int weightDim,
                                                  int contextLength,
                                                  int contextStart,
                                                  int beginPad) {
  __shared__ real sum_s[THREADS_Y][THREADS_X];
  int padOfBlock = (weightDim + THREADS_X - 1) / THREADS_X;
  const int idx = threadIdx.x;
  const int idy = threadIdx.y;
  int padId = blockIdx.x / padOfBlock;
  int weightIdx = idx + THREADS_X * (blockIdx.x % padOfBlock);
  int instanceId;
  real value = 0;
  real* output_r;

  sum_s[idy][idx] = 0.0f;
  if (weightIdx < weightDim) {
    for (int seqId = idy; seqId < numSequences; seqId += THREADS_Y) {
      int seqStart = sequence[seqId];
      int seqEnd = sequence[seqId+1];
      output_r = outputGrad + seqStart * weightDim * contextLength;

      if (contextStart < 0) {
        if (padId + contextStart < 0) {
          instanceId = padId;
        } else {
          // beginPad > 0;
          instanceId = (padId - beginPad) + (seqEnd - seqStart) - contextStart;
        }
      } else {
        if (padId + (seqEnd - seqStart) < contextStart) {
          continue;
        } else {
          // beginPad == 0;
          instanceId = padId + (seqEnd - seqStart) - contextStart;
        }
      }

      int outx = (instanceId - contextLength) < 0 ?
                 instanceId : (contextLength - 1);
      int outy = (instanceId - contextLength) < 0 ?
                 0 : (instanceId - (contextLength - 1));
      output_r += outy * weightDim * contextLength + outx * weightDim;
      for (int j = outy; j < seqEnd - seqStart; j++) {
        value += output_r[weightIdx];
        if (j - outy == outx) break;
        output_r += (contextLength - 1) * weightDim;
      }
    }
    sum_s[idy][idx] = value;
  }
  __syncthreads();

  for (int stride = THREADS_Y/2; stride > 0; stride = stride/2) {
    if (idy < stride) {
      sum_s[idy][idx] += sum_s[idy + stride][idx];
    }
    __syncthreads();
  }
  __syncthreads();

  if (weightIdx < weightDim) {
    if (idy == 0) {
      weightGrad[padId * weightDim + weightIdx] += sum_s[0][idx];
    }
  }
}

void hl_context_projection_backward_weight(real* outputGrad,
                                           const int* sequence,
                                           real* weightGrad,
                                           int numSequences,
                                           int weightDim,
                                           int totalPad,
                                           int contextLength,
                                           int contextStart,
                                           int beginPad) {
  CHECK_NOTNULL(outputGrad);
  CHECK_NOTNULL(sequence);
  CHECK_NOTNULL(weightGrad);

  int threadsX = 32;
  int threadsY = 32;
  int blocksX = totalPad * ((weightDim + threadsX - 1) / threadsX);
  dim3 threads(threadsX, threadsY);
  dim3 grid(blocksX, 1);

  KeContextProjectionBackwardWeight<32, 32>
    <<< grid, threads, 0, STREAM_DEFAULT >>>
    (outputGrad, sequence, weightGrad, numSequences, weightDim,
     contextLength, contextStart, beginPad);
  CHECK_SYNC("hl_context_projection_backward_weight failed");
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
    for (int i = 0; i < seqLength; i++) {
      sum += src[(start + i) * width + col];
    }
    sum = mode == 1 ? sum :
        (mode == 0 ? sum / seqLength : sum * my_rsqrt((real)seqLength));
    dst[row * width + col] = sum;
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
