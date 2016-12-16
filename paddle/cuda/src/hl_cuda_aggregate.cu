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
#include "hl_cuda.h"
#include "hl_cuda.ph"
#include "hl_aggregate.h"
#include "hl_thread.ph"
#include "hl_matrix_base.cuh"
#include "paddle/utils/Logging.h"

/**
 * @brief   matrix row operator.
 */
template<class Agg, int blockSize>
__global__ void KeMatrixRowOp(Agg agg,
                              real *E,
                              real *Sum,
                              int dimN) {
  __shared__ real sum_s[blockSize];
  int cnt = (dimN + blockSize -1) / blockSize;
  int rowId = blockIdx.x + blockIdx.y*gridDim.x;
  int index = rowId*dimN;
  int tid = threadIdx.x;
  int lmt = tid;

  real tmp = agg.init();
  for (int ii = 0; ii < cnt && lmt < dimN; ii++) {
    tmp = agg(tmp, E[index + lmt]);
    lmt += blockSize;
  }
  sum_s[tid] = tmp;
  __syncthreads();

  for (int stride = blockSize/2; stride > 0; stride = stride/2) {
    if (tid < stride) {
      sum_s[tid] = agg(sum_s[tid], sum_s[tid + stride]);
    }
    __syncthreads();
  }
  __syncthreads();

  if (tid == 0) {
    Sum[rowId] = sum_s[0];
  }
}

template <class Agg>
void hl_matrix_row_op(Agg agg,
                      real *A_d,
                      real *C_d,
                      int dimM,
                      int dimN) {
  int blocksX = dimM;
  int blocksY = 1;
  dim3 threads(128, 1);
  dim3 grid(blocksX, blocksY);

  KeMatrixRowOp<Agg, 128><<< grid, threads, 0, STREAM_DEFAULT >>>
           (agg, A_d, C_d, dimN);
}

void hl_matrix_row_sum(real *A_d, real *C_d, int dimM, int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  hl_matrix_row_op(aggregate::sum(),
                   A_d,
                   C_d,
                   dimM,
                   dimN);
  CHECK_SYNC("hl_matrix_row_sum failed");
}

void hl_matrix_row_max(real *A_d, real *C_d, int dimM, int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  hl_matrix_row_op(aggregate::max(),
                   A_d,
                   C_d,
                   dimM,
                   dimN);
  CHECK_SYNC("hl_matrix_row_max failed");
}

void hl_matrix_row_min(real *A_d, real *C_d, int dimM, int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  hl_matrix_row_op(aggregate::min(),
                   A_d,
                   C_d,
                   dimM,
                   dimN);
  CHECK_SYNC("hl_matrix_row_min failed");
}

/**
 * @brief   matrix column operator.
 */
template<class Agg>
__global__ void KeMatrixColumnOp(Agg agg,
                                 real *E,
                                 real *Sum,
                                 int dimM,
                                 int dimN) {
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  real tmp = agg.init();
  if (rowIdx < dimN) {
    for (int index = 0; index < dimM; index++) {
      tmp = agg(tmp, E[dimN * index + rowIdx]);
    }
    Sum[rowIdx] = tmp;
  }
}

template<class Agg, int blockDimX, int blockDimY>
__global__ void KeMatrixColumnOp_S(Agg agg,
                                   real *E,
                                   real *Sum,
                                   int dimM,
                                   int dimN) {
    __shared__ real _sum[blockDimX*blockDimY];
    int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = threadIdx.y;

  real tmp = agg.init();
  if (rowIdx < dimN) {
    for (; index < dimM;) {
      tmp = agg(tmp, E[dimN * index + rowIdx]);
      index += blockDimY;
    }
  }
  _sum[threadIdx.x + threadIdx.y*blockDimX] = tmp;
  __syncthreads();

  if (rowIdx < dimN) {
    if (threadIdx.y ==0) {
      real tmp = agg.init();
      for (int i=0; i < blockDimY; i++) {
        tmp = agg(tmp, _sum[threadIdx.x + i*blockDimX]);
      }
      Sum[rowIdx] = tmp;
    }
  }
}

template <class Agg>
void hl_matrix_column_op(Agg agg,
                         real *A_d,
                         real *C_d,
                         int dimM,
                         int dimN) {
  if (dimN >= 8192) {
    int blocksX = (dimN + 128 -1) / 128;
    int blocksY = 1;
    dim3 threads(128, 1);
    dim3 grid(blocksX, blocksY);
    KeMatrixColumnOp<Agg><<< grid, threads, 0, STREAM_DEFAULT >>>
             (agg, A_d, C_d, dimM, dimN);
  } else {
    int blocksX = (dimN + 32 -1) / 32;
    int blocksY = 1;
    dim3 threads(32, 32);
    dim3 grid(blocksX, blocksY);
    KeMatrixColumnOp_S<Agg, 32, 32><<< grid, threads, 0, STREAM_DEFAULT>>>
             (agg, A_d, C_d, dimM, dimN);
  }

  return;
}

void hl_matrix_column_sum(real *A_d, real *C_d, int dimM, int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  hl_matrix_column_op(aggregate::sum(),
                      A_d,
                      C_d,
                      dimM,
                      dimN);

  CHECK_SYNC("hl_matrix_column_sum failed");
}

void hl_matrix_column_max(real *A_d, real *C_d, int dimM, int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  hl_matrix_column_op(aggregate::max(),
                      A_d,
                      C_d,
                      dimM,
                      dimN);

  CHECK_SYNC("hl_matrix_column_max failed");
}

void hl_matrix_column_min(real *A_d, real *C_d, int dimM, int dimN) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_d);

  hl_matrix_column_op(aggregate::min(),
                      A_d,
                      C_d,
                      dimM,
                      dimN);

  CHECK_SYNC("hl_matrix_column_min failed");
}

template <int blockSize>
__global__ void KeVectorSum(real *E, real *Sum, int dimM) {
  __shared__ double sum_s[blockSize];
  int tid = threadIdx.x;
  int index = blockIdx.y*blockDim.x+threadIdx.x;

  sum_s[tid] = 0.0f;
  while (index < dimM) {
    sum_s[tid] += E[index];
    index += blockDim.x*gridDim.y;
  }
  __syncthreads();

  for (int stride = blockSize/2; stride > 0; stride = stride/2) {
    if (tid < stride) {
      sum_s[tid] += sum_s[tid + stride];
    }
    __syncthreads();
  }
  __syncthreads();

  if (tid == 0) {
    Sum[blockIdx.y] = sum_s[0];
  }
}

void hl_vector_sum(real *A_d, real *C_h, int dimM) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_h);

  int blockSize = 128;
  int gridSize = 128;
  int blocksX = 1;
  int blocksY = gridSize;
  dim3 threads(blockSize, 1);
  dim3 grid(blocksX, blocksY);

  struct _hl_event_st hl_event_st  = {.cu_event = t_resource.event};
  hl_event_t hl_event = &hl_event_st;
  while (!hl_cuda_event_is_ready(hl_event)) {}

  KeVectorSum<128><<< grid, threads, 0, STREAM_DEFAULT >>>
           (A_d, t_resource.gpu_mem, dimM);
  KeVectorSum<128><<< 1, threads, 0, STREAM_DEFAULT >>>
           (t_resource.gpu_mem, t_resource.cpu_mem, 128);

  hl_memcpy_async(C_h, t_resource.cpu_mem, sizeof(real), HPPL_STREAM_DEFAULT);
  hl_stream_record_event(HPPL_STREAM_DEFAULT, hl_event);

  hl_stream_synchronize(HPPL_STREAM_DEFAULT);
  cudaError_t err = (cudaError_t)hl_get_device_last_error();
  CHECK_EQ(cudaSuccess, err)
    << "CUDA error: " << hl_get_device_error_string((size_t)err);
}

template <int blockSize>
__global__ void KeVectorAbsSum(real *E, real *Sum, int dimM) {
  __shared__ double sum_s[blockSize];
  int tid = threadIdx.x;
  int index = blockIdx.y*blockDim.x+threadIdx.x;

  sum_s[tid] = 0.0f;
  while (index < dimM) {
    sum_s[tid] += abs(E[index]);
    index += blockDim.x*gridDim.y;
  }
  __syncthreads();

  for (int stride = blockSize/2; stride > 0; stride = stride/2) {
    if (tid < stride) {
      sum_s[tid] += sum_s[tid + stride];
    }
    __syncthreads();
  }
  __syncthreads();

  if (tid == 0) {
    Sum[blockIdx.y] = sum_s[0];
  }
}

void hl_vector_abs_sum(real *A_d, real *C_h, int dimM) {
  CHECK_NOTNULL(A_d);
  CHECK_NOTNULL(C_h);

  int blockSize = 128;
  int gridSize = 128;
  int blocksX = 1;
  int blocksY = gridSize;
  dim3 threads(blockSize, 1);
  dim3 grid(blocksX, blocksY);

  struct _hl_event_st hl_event_st  = {.cu_event = t_resource.event};
  hl_event_t hl_event = &hl_event_st;
  while (!hl_cuda_event_is_ready(hl_event)) {}

  KeVectorAbsSum<128><<< grid, threads, 0, STREAM_DEFAULT >>>
           (A_d, t_resource.gpu_mem, dimM);
  KeVectorAbsSum<128><<< 1, threads, 0, STREAM_DEFAULT >>>
           (t_resource.gpu_mem, t_resource.cpu_mem, 128);

  hl_memcpy_async(C_h, t_resource.cpu_mem, sizeof(real), HPPL_STREAM_DEFAULT);
  hl_stream_record_event(HPPL_STREAM_DEFAULT, hl_event);

  hl_stream_synchronize(HPPL_STREAM_DEFAULT);
  cudaError_t err = (cudaError_t)hl_get_device_last_error();
  CHECK_EQ(cudaSuccess, err)
    << "CUDA error: " << hl_get_device_error_string((size_t)err);
}
