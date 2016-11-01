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
#include "hl_matrix.h"
#include "hl_matrix_ops.cuh"
#include "hl_matrix_apply.cuh"
#include "hl_sequence.h"
#include "paddle/utils/Logging.h"
#include "hl_device_functions.cuh"

DEFINE_MATRIX_UNARY_OP(Zero, a = 0);
DEFINE_MATRIX_TERNARY_PARAMETER_OP(_add, TWO_PARAMETER, c = p1*a + p2*b);
void hl_matrix_add(real *A_d,
                   real *B_d,
                   real *C_d,
                   int dimM,
                   int dimN,
                   real alpha,
                   real beta) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);
  CHECK_NOTNULL(C_d);

  hl_gpu_apply_ternary_op
    <real, ternary::_add<real>, 0, 0>(ternary::_add<real>(alpha, beta),
                                      A_d,
                                      B_d,
                                      C_d,
                                      dimM,
                                      dimN,
                                      dimN,
                                      dimN,
                                      dimN);
  CHECK_SYNC("hl_matrix_add failed");
}

#ifdef PADDLE_TYPE_DOUBLE
    #define THRESHOLD   128
#else
    #define THRESHOLD   64
#endif
__device__ __forceinline__
void findMax(real* I,
             real* dfMax_s,
             int blockSize,
             int base,
             int curIdx,
             int nextIdx,
             int dimN,
             real* max) {
  dfMax_s[base] = -1.0e20;
  while (curIdx < dimN) {
    if (dfMax_s[base] < I[nextIdx]) {
      dfMax_s[base] = I[nextIdx];
    }
    nextIdx += blockSize;
    curIdx += blockSize;
  }
  __syncthreads();

  for (int stride = blockSize >> 1; stride > 0; stride >>= 1) {
    __syncthreads();
    if (base < stride) {
      nextIdx = base + stride;
      if (dfMax_s[base] < dfMax_s[nextIdx]) {
          dfMax_s[base] = dfMax_s[nextIdx];
      }
    }
  }

  if (0 == base)  {
    max[0] = dfMax_s[0];
  }
  __syncthreads();
}

__device__ __forceinline__
void subMaxAndExp(real* I,
                  real* O,
                  int curIdx,
                  int nextIdx,
                  int blockSize,
                  int dimN,
                  real max) {
  real val;
  while (curIdx < dimN) {
    val = I[nextIdx] - max;
    if (val < -THRESHOLD) {
      val = -THRESHOLD;
    }
    I[nextIdx] = val;
#ifndef PADDLE_TYPE_DOUBLE
    O[nextIdx] = __expf(val);
#else
    O[nextIdx] = exp(val);
#endif
    nextIdx += blockSize;
    curIdx += blockSize;
  }
  __syncthreads();
}

__device__ __forceinline__
void valueSum(real* O,
              real* dfMax_s,
              int blockSize,
              int base,
              int curIdx,
              int nextIdx,
              int dimN) {
  dfMax_s[base] = 0;
  while (curIdx < dimN) {
    dfMax_s[base] += O[nextIdx];
    nextIdx += blockSize;
    curIdx += blockSize;
  }
  __syncthreads();

  for (int stride = blockSize >> 1; stride > 0; stride >>= 1) {
    __syncthreads();
    if (base < stride) {
      nextIdx = base + stride;
      dfMax_s[base] += dfMax_s[nextIdx];
    }
  }
  __syncthreads();
}

__device__ __forceinline__
void divSum(real* O,
            real sum,
            int curIdx,
            int nextIdx,
            int blockSize,
            int dimN) {
  while (curIdx < dimN) {
    O[nextIdx] /= sum;
    nextIdx += blockSize;
    curIdx += blockSize;
  }
}

__device__ __forceinline__
void softmax(real* I,
             real* O,
             real* dfMax_s,
             int blockSize,
             int base,
             int curIdx,
             int nextIdx,
             int dimN) {
  __shared__ real max;

  // find the max number
  findMax(I, dfMax_s, blockSize, base, curIdx,
          nextIdx, dimN, &max);

  // sub max Value and do Exp operation
  subMaxAndExp(I, O, base, nextIdx, blockSize, dimN, max);

  // add dimN values into blockDim.x buffer
  // sum is in dfMax_s[0]
  valueSum(O, dfMax_s, blockSize, base, curIdx, nextIdx, dimN);

  // divided by sum
  divSum(O, dfMax_s[0], curIdx, nextIdx, blockSize, dimN);
}

template<int blockSize>
__global__ void KeMatrixSoftMax(real *O, real *I, int dimN) {
  int base = threadIdx.x;
  __shared__ real dfMax_s[blockSize];
  int nextIdx = blockIdx.x * dimN + base;
  int curIdx = base;

  softmax(I, O, dfMax_s, blockSize, base, curIdx, nextIdx, dimN);
}

void hl_matrix_softmax(real *A_d, real *C_d, int dimM, int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  dim3 block(512, 1);
  dim3 grid(dimM, 1);
  KeMatrixSoftMax<512>
           <<<grid, block, 0, STREAM_DEFAULT>>>(C_d, A_d, dimN);
  CHECK_SYNC("hl_matrix_softmax failed");
}

template<int blockSize>
__global__ void KeSequenceSoftMax(real *O, real *I, const int* index) {
  int base = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ real dfMax_s[blockSize];

  int start = index[bid];
  int dimN = index[bid + 1] - start;

  int nextIdx = start + base;
  int curIdx = base;

  softmax(I, O, dfMax_s, blockSize, base, curIdx, nextIdx, dimN);
}

void hl_sequence_softmax_forward(real *A_d,
                                 real *C_d,
                                 const int* index,
                                 int numSequence) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  dim3 block(512, 1);
  dim3 grid(numSequence, 1);
  KeSequenceSoftMax<512>
           <<<grid, block, 0, STREAM_DEFAULT>>>(C_d, A_d, index);
  CHECK_SYNC("hl_sequence_softmax_forward failed");
}

__global__ void KeMatrixDerivative(real *grad_d,
                                   real *output_d,
                                   real *sftmaxSum_d,
                                   int dimM,
                                   int dimN) {
  int rowIdx = blockIdx.x*blockDim.x + threadIdx.x;
  int colIdx = blockIdx.y*blockDim.y + threadIdx.y;
  int index;

  if (rowIdx < dimM && colIdx < dimN) {
    index = rowIdx*dimN + colIdx;
    grad_d[index] = output_d[index] * (grad_d[index] - sftmaxSum_d[rowIdx]);
  }
}

void hl_matrix_softmax_derivative(real *grad_d,
                                  real *output_d,
                                  real *sftmaxSum_d,
                                  int dimM,
                                  int dimN) {
  CHECK_NOTNULL(grad_d);
  CHECK_NOTNULL(output_d);
  CHECK_NOTNULL(sftmaxSum_d);

  int blocksX = (dimM + 0) / 1;
  int blocksY = (dimN + 1024 -1) / 1024;
  dim3 threads(1, 1024);
  dim3 grid(blocksX, blocksY);

  KeMatrixDerivative<<< grid, threads, 0, STREAM_DEFAULT >>>
           (grad_d, output_d, sftmaxSum_d, dimM, dimN);
  CHECK_SYNC("hl_matrix_softmax_derivative failed");
}

template<int blockSize>
__global__ void KeMatrixClassificationError(real* in_A,
                                            int* in_B,
                                            real* out_C,
                                            int dimN) {
  __shared__ real max_s[blockSize];
  __shared__ int max_l[blockSize];
  const int tid = threadIdx.x;
  const int rowId = blockIdx.x;

  max_s[tid] = -1e30f;
  in_A += rowId * dimN;
  real tmp;
  for (int colId = tid; colId < dimN; colId += blockSize) {
    tmp = in_A[colId];
    if (max_s[tid] < tmp) {
      max_s[tid] = tmp;
      max_l[tid] = colId;
    }
  }
  __syncthreads();

  for (int stride = blockSize/2; stride > 0; stride = stride/2) {
    if (tid < stride) {
      if (max_s[tid] < max_s[tid + stride]) {
        max_s[tid] = max_s[tid + stride];
        max_l[tid] = max_l[tid + stride];
      }
    }
    __syncthreads();
  }
  __syncthreads();

  if (tid == 0) {
    out_C[rowId] = (max_l[0] == in_B[rowId] ? 0 : 1.0f);
  }
}

void hl_matrix_classification_error(real* A_d,
                                    int* B_d,
                                    real* C_d,
                                    int dimM,
                                    int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(B_d);
  CHECK_NOTNULL(C_d);

  // each sample is calculated by one block
  KeMatrixClassificationError<1024><<< dimM, 1024, 0, STREAM_DEFAULT >>>
    (A_d, B_d, C_d, dimN);
  CHECK_SYNC("hl_matrix_classification_error");
}

__global__ void KeMatrixCrossEntropy(real* O,
                                     real* E,
                                     int* label,
                                     int dimM,
                                     int dimN) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int newBase;
  if (index < dimM) {
    newBase = label[index];
    newBase = newBase % dimN;
    E[index] = -log(O[index * dimN + newBase]);
  }
}

void hl_matrix_cross_entropy(real* A_d,
                             real* C_d,
                             int* label_d,
                             int dimM,
                             int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  int blocks = (dimM + 1024 - 1) / 1024;
  dim3 threads(1024, 1);
  dim3 grid(blocks, 1);
  KeMatrixCrossEntropy<<< grid, threads, 0, STREAM_DEFAULT >>>
           (A_d, C_d, label_d, dimM, dimN);
  CHECK_SYNC("hl_matrix_cross_entropy failed");
}

__global__ void KeMatrixCrossEntropyBp(real* grad_d,
                                       real* output_d,
                                       int* label_d,
                                       int dimM,
                                       int dimN) {
  int rowIdx = blockIdx.x*blockDim.x + threadIdx.x;
  int colIdx = blockIdx.y*blockDim.y + threadIdx.y;
  int index;
  if (rowIdx < dimM && colIdx < dimN) {
    index = rowIdx*dimN + colIdx;
    if (label_d[rowIdx] == colIdx) {
      grad_d[index] -= 1.0f / output_d[index];
    }
  }
}

void hl_matrix_cross_entropy_bp(real* grad_d,
                                real* output_d,
                                int* label_d,
                                int dimM,
                                int dimN) {
  CHECK_NOTNULL(grad_d);
  CHECK_NOTNULL(output_d);
  CHECK_NOTNULL(label_d);

  int blocksX = (dimM + 0)/1;
  int blocksY = (dimN + 1024 -1) / 1024;
  dim3 threads(1, 1024);
  dim3 grid(blocksX, blocksY);
  KeMatrixCrossEntropyBp<<< grid, threads, 0, STREAM_DEFAULT >>>
           (grad_d, output_d, label_d, dimM, dimN);
  CHECK_SYNC("hl_matrix_cross_entropy_bp failed");
}

void hl_matrix_zero_mem(real* data, int num) {
  hl_gpu_apply_unary_op(
        unary::Zero<real>(), data, 1, num, num);
}

__global__ void KeParamReluForward(real* output,
                                   real* input,
                                   real* w,
                                   int width,
                                   int height,
                                   int partial_sum) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx < width && ty < height) {
    int index = ty * width + tx;
    output[index] = input[index] > 0 ? input[index] :
        input[index] * w[tx / partial_sum];
  }
}

void hl_param_relu_forward(real* output,
                           real* input,
                           real* w,
                           int width,
                           int height,
                           int partial_sum) {
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(input);
  CHECK_NOTNULL(w);
  dim3 threads(16, 16);
  int blockX = (width + 16 - 1) / 16;
  int blockY = (height + 16 -1) / 16;
  dim3 grid(blockX, blockY);
  KeParamReluForward<<<grid, threads, 0, STREAM_DEFAULT>>>
    (output, input, w, width, height, partial_sum);
  CHECK_SYNC("hl_param_relu_forward failed");
}

template<int blockSize>
__global__ void KeParamReluBackWardW(real* grad_w,
                                     real* grad_o,
                                     real* input,
                                     int width,
                                     int height,
                                     int partial_sum) {
  const int tid = threadIdx.x;
  __shared__ real temp[blockSize];
  grad_o += partial_sum * blockIdx.x;
  input += partial_sum * blockIdx.x;
  real tmp = 0.0;
  for (int index = tid; index < partial_sum * height; index += blockSize) {
    int row = index / partial_sum;
    int offset = row * width + (index - row * partial_sum);
    if (input[offset] < 0) {
      tmp += grad_o[offset] * input[offset];
    }
  }
  temp[tid] = tmp;
  __syncthreads();
  for (int s = blockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      temp[tid] += temp[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    grad_w[blockIdx.x] += temp[0];
  }
}

void hl_param_relu_backward_w(real* grad_w,
                              real* grad_o,
                              real* input,
                              int width,
                              int height,
                              int partial_sum) {
  CHECK_NOTNULL(grad_w);
  CHECK_NOTNULL(grad_o);
  CHECK_NOTNULL(input);
  const int blockSize = 1024;
  int grid_num = width / partial_sum;
  dim3 threads(blockSize, 1);
  dim3 grid(grid_num, 1);
  KeParamReluBackWardW<blockSize><<<grid, threads, 0, STREAM_DEFAULT>>>
    (grad_w, grad_o, input, width, height, partial_sum);
  CHECK_SYNC("hl_param_relu_backward_w failed");
}

__global__ void KeParamReluBackwardDiff(real* grad_o,
                                        real* input,
                                        real* w,
                                        real* diff,
                                        int width,
                                        int height,
                                        int partial_sum) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx < width && ty < height) {
    int index = ty * width + tx;
    diff[index] += grad_o[index] * (input[index] > 0 ? 1 : w[tx / partial_sum]);
  }
}

void hl_param_relu_backward_diff(real* grad_o,
                                 real* data,
                                 real* w,
                                 real* diff,
                                 int width,
                                 int height,
                                 int partial_sum) {
  CHECK_NOTNULL(grad_o);
  CHECK_NOTNULL(data);
  CHECK_NOTNULL(w);
  CHECK_NOTNULL(diff);
  dim3 threads(16, 16);
  int blockX = (width + 16 - 1) / 16;
  int blockY = (height + 16 -1) / 16;
  dim3 grid(blockX, blockY);
  KeParamReluBackwardDiff<<<grid, threads, 0, STREAM_DEFAULT>>>
      (grad_o, data, w, diff, width, height, partial_sum);
  CHECK_SYNC("hl_param_relu_backward_diff failed");
}

template<int blockSize>
__global__ void KeCosSim(real* output,
                         real* input1,
                         real* input2,
                         int width,
                         int input1_height,
                         int input2_height,
                         real scale) {
  const int ty = blockIdx.y;
  int tid = threadIdx.x;

  __shared__ real xx[blockSize];
  __shared__ real yy[blockSize];
  __shared__ real xy[blockSize];

  xx[tid] = 0.0;
  yy[tid] = 0.0;
  xy[tid] = 0.0;
  __syncthreads();

  input1 += ty * width;
  if (input2_height > 1) {
    input2 += ty * width;
  }
  for (int index = tid; index < width; index += blockSize) {
    real x = input1[index];
    real y = input2[index];
    xx[tid] += x * x;
    yy[tid] += y * y;
    xy[tid] += x * y;
  }
  __syncthreads();

  for (int s = blockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      xx[tid] += xx[tid + s];
      yy[tid] += yy[tid + s];
      xy[tid] += xy[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[ty] = scale * xy[0] / (sqrt(xx[0]) * sqrt(yy[0]));
  }
}

void hl_cossim(real* output,
               real* input1,
               real* input2,
               int width,
               int input1_height,
               int input2_height,
               real scale) {
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(input1);
  CHECK_NOTNULL(input2);
  const int blockSize = 256;
  dim3 threads(blockSize, 1);
  dim3 grid(1, input1_height);

  KeCosSim<blockSize><<<grid, threads, 0, STREAM_DEFAULT>>>
    (output, input1, input2, width, input1_height, input2_height, scale);
  CHECK_SYNC("hl_cossim failed");
}

template<int blockSize>
__global__ void KeCosSimDerivative(real* grad,
                                   real* output,
                                   real* prevOutX,
                                   real* prevOutY,
                                   real* prevGradX,
                                   real* prevGradY,
                                   int width,
                                   int input1_height,
                                   int input2_height,
                                   real scale) {
  const int ty = blockIdx.y;
  int tid = threadIdx.x;

  __shared__ real xx[blockSize];
  __shared__ real yy[blockSize];
  __shared__ real xy[blockSize];

  xx[tid] = 0.0;
  yy[tid] = 0.0;
  xy[tid] = 0.0;
  __syncthreads();

  prevOutX += ty * width;
  prevGradX += ty * width;
  if (input2_height > 1) {
    prevOutY += ty * width;
    prevGradY += ty * width;
  }
  for (int index = tid; index < width; index += blockSize) {
    real x = prevOutX[index];
    real y = prevOutY[index];
    xx[tid] += x * x;
    yy[tid] += y * y;
    xy[tid] += x * y;
  }
  __syncthreads();

  for (int s = blockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      xx[tid] += xx[tid + s];
      yy[tid] += yy[tid + s];
      xy[tid] += xy[tid + s];
    }
    __syncthreads();
  }
  if (xy[0] == 0) {
    real reciprocal = 1.0 / (sqrt(xx[0]) * sqrt(yy[0]));
    for (int index = tid; index < width; index += blockSize) {
      prevGradX[index] +=
        scale * grad[ty] * prevOutY[index] * reciprocal;
      if (input2_height > 1) {
        prevGradY[index] +=
          scale * grad[ty] * prevOutX[index] * reciprocal;
      } else {
        paddle::paddleAtomicAdd(prevGradY + index,
          scale * grad[ty] * prevOutX[index] * reciprocal);
      }
    }
  } else {
    real reciprocalXY = 1.0 / xy[0];
    real reciprocalSquareSumX = 1.0 / xx[0];
    real reciprocalSquareSumY = 1.0 / yy[0];
    for (int index = tid; index < width; index += blockSize) {
      prevGradX[index] += output[ty] * grad[ty] *
        (prevOutY[index] * reciprocalXY -
         prevOutX[index] * reciprocalSquareSumX);
      if (input2_height > 1) {
        prevGradY[index] += output[ty] * grad[ty] *
          (prevOutX[index] * reciprocalXY -
           prevOutY[index] * reciprocalSquareSumY);
      } else {
        paddle::paddleAtomicAdd(prevGradY + index, output[ty] * grad[ty] *
          (prevOutX[index] * reciprocalXY -
           prevOutY[index] * reciprocalSquareSumY));
      }
    }
  }
}


void hl_cossim_derivative(real* grad,
                          real* output,
                          real* prevOutX,
                          real* prevOutY,
                          real* prevGradX,
                          real* prevGradY,
                          int width,
                          int input1_height,
                          int input2_height,
                          real scale) {
  CHECK_NOTNULL(grad);
  CHECK_NOTNULL(output);
  CHECK_NOTNULL(prevOutX);
  CHECK_NOTNULL(prevOutY);
  CHECK_NOTNULL(prevGradX);
  CHECK_NOTNULL(prevGradY);
  const int blockSize = 256;
  dim3 threads(blockSize, 1);
  dim3 grid(1, input1_height);
  KeCosSimDerivative<blockSize><<<grid, threads, 0, STREAM_DEFAULT>>>
    (grad, output, prevOutX, prevOutY, prevGradX, prevGradY, width,
        input1_height, input2_height, scale);
  CHECK_SYNC("hl_cossim_derivate failed");
}
