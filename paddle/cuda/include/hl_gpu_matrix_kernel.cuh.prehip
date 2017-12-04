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



#ifndef HL_GPU_MATRIX_KERNEL_CUH_
#define HL_GPU_MATRIX_KERNEL_CUH_

#include <algorithm>
#include "paddle/utils/Logging.h"
#include "hl_base.h"

#ifdef __NVCC__
/* gpu apply interface */

template<class T, class Op>
__global__ void KeEltWiseUnaryOp(T* A_d, const int border, Op op) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < border) {
    op.gpuOperator(A_d[idx]);
  }
}

template<class T, class Op>
__global__ void KeEltWiseUnaryOp(T* A_d,
                                 int dimM,
                                 int dimN,
                                 int lda,
                                 Op op) {
  const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = rowIdx; i < dimM; i += gridDim.y * blockDim.y) {
    for (int j = colIdx; j < dimN; j += gridDim.x * blockDim.x) {
      op.gpuOperator(A_d[i * lda + j]);
    }
  }
}

template<class T, class Op>
__global__ void KeEltWiseBinaryOp(T* A_d, T *B_d, const int border, Op op) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < border) {
    op.gpuOperator(A_d[idx], B_d[idx]);
  }
}

template<class T, class Op, bool BAsRowVector, bool BAsColVector>
__global__ void KeEltWiseBinaryOp(T *A_d,
                                  T *B_d,
                                  int dimM,
                                  int dimN,
                                  int lda,
                                  int ldb,
                                  Op op) {
  const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = rowIdx; i < dimM; i += gridDim.y * blockDim.y) {
    for (int j = colIdx; j < dimN; j += gridDim.x * blockDim.x) {
      if (BAsRowVector == 0 && BAsColVector == 0) {
        op.gpuOperator(A_d[i * lda + j], B_d[i * ldb + j]);
      } else if (BAsRowVector == 1 && BAsColVector == 0) {
        op.gpuOperator(A_d[i * lda + j], B_d[j]);
      } else if (BAsRowVector == 0 && BAsColVector == 1) {
        op.gpuOperator(A_d[i * lda + j], B_d[i * ldb]);
      } else {
        op.gpuOperator(A_d[i * lda + j], B_d[0]);
      }
    }
  }
}

template<class T, class Op>
__global__ void KeEltWiseTernaryOp(T* A_d,
                                   T *B_d,
                                   T *C_d,
                                   const int border,
                                   Op op) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < border) {
    op.gpuOperator(A_d[idx], B_d[idx], C_d[idx]);
  }
}

template<class T, class Op, bool CAsRowVector, bool CAsColVector>
__global__ void KeEltWiseTernaryOp(T* A_d,
                                   T* B_d,
                                   T* C_d,
                                   int dimM,
                                   int dimN,
                                   int lda,
                                   int ldb,
                                   int ldc,
                                   Op op) {
  const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = rowIdx; i < dimM; i += gridDim.y * blockDim.y) {
    for (int j = colIdx; j < dimN; j += gridDim.x * blockDim.x) {
      if (CAsRowVector == 0 && CAsColVector == 0) {
        op.gpuOperator(A_d[i*lda + j], B_d[i*ldb + j], C_d[i*ldc + j]);
      } else if (CAsRowVector == 1 && CAsColVector == 0) {
        op.gpuOperator(A_d[i*lda + j], B_d[i*ldb + j], C_d[j]);
      } else if (CAsRowVector == 0 && CAsColVector == 1) {
        op.gpuOperator(A_d[i*lda + j], B_d[i*ldb + j], C_d[i*ldc]);
      } else {
        op.gpuOperator(A_d[i*lda + j], B_d[i*ldb + j], C_d[0]);
      }
    }
  }
}

template<class T, class Op>
__global__ void KeEltWiseQuaternaryOp(T* A_d,
                                      T* B_d,
                                      T* C_d,
                                      T* D_d,
                                      const int border,
                                      Op op) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < border) {
    op.gpuOperator(A_d[idx], B_d[idx], C_d[idx], D_d[idx]);
  }
}

template<class T, class Op>
__global__ void KeEltWiseQuaternaryOp(T* A_d,
                                      T* B_d,
                                      T* C_d,
                                      T* D_d,
                                      int dimM,
                                      int dimN,
                                      int lda,
                                      int ldb,
                                      int ldc,
                                      int ldd,
                                      Op op) {
  const int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
  for (int i = rowIdx; i < dimM; i += gridDim.y * blockDim.y) {
    for (int j = colIdx; j < dimN; j += gridDim.x * blockDim.x) {
      op.gpuOperator(A_d[i*lda + j],
        B_d[i*ldb + j], C_d[i*ldc + j], D_d[i*ldd + j]);
    }
  }
}

/**
 * @brief   gpu element wise unary operator.
 */
template <class T, class Op>
void hl_gpu_apply_unary_op(Op op, T* A_d, int dimM, int dimN, int lda) {
  CHECK_NOTNULL(A_d);

  if (dimM == 1 || dimN == lda) {
    int size = dimM * dimN;
    int blockSize = size <= 1024 ? size : 1024;
    int gridSize = (size + 1024 - 1) / 1024;
    KeEltWiseUnaryOp<T, Op><<<gridSize, blockSize, 0, STREAM_DEFAULT>>>
      (A_d, size, op);
  } else {
    int blockSizeY = std::min(32, dimM);
    int blockSizeX = (32 / blockSizeY) * 32;
    int gridSizeX = std::min(32, (dimN + blockSizeX - 1) / blockSizeX);
    int gridSizeY = std::min(32, (dimM + blockSizeY - 1) / blockSizeY);
    dim3 threads(blockSizeX, blockSizeY);
    dim3 grid(gridSizeX, gridSizeY);
    KeEltWiseUnaryOp<T, Op><<<grid, threads, 0, STREAM_DEFAULT>>>
      (A_d, dimM, dimN, lda, op);
  }

  CHECK_SYNC("hl_gpu_apply_unary_op failed");
}

/**
 * @brief   gpu element wise binary operator.
 */
template <class T, class Op, bool BAsRowVector, bool BAsColVector>
void hl_gpu_apply_binary_op(Op op,
                            T* A_d,
                            T* B_d,
                            int dimM,
                            int dimN,
                            int lda,
                            int ldb) {
  CHECK_NOTNULL(A_d);

  if ((BAsRowVector == 0 && BAsColVector == 0) &&
      ((dimM == 1) || (dimN == lda && dimN == ldb))) {
    int size = dimM * dimN;
    int blockSize = size <= 1024 ? size : 1024;
    int gridSize = (size + 1024 - 1) / 1024;
    KeEltWiseBinaryOp<T, Op><<<gridSize, blockSize, 0, STREAM_DEFAULT>>>
      (A_d, B_d, size, op);
  } else {
    int blockSizeY = std::min(32, dimM);
    int blockSizeX = (32 / blockSizeY) * 32;
    int gridSizeX = std::min(32, (dimN + blockSizeX - 1) / blockSizeX);
    int gridSizeY = std::min(32, (dimM + blockSizeY - 1) / blockSizeY);
    dim3 threads(blockSizeX, blockSizeY);
    dim3 grid(gridSizeX, gridSizeY);
    KeEltWiseBinaryOp<T, Op, BAsRowVector, BAsColVector>
      <<<grid, threads, 0, STREAM_DEFAULT>>>
      (A_d, B_d, dimM, dimN, lda, ldb, op);
  }

  CHECK_SYNC("hl_gpu_apply_binary_op failed");
}

/**
 * @brief   gpu element wise ternary operator.
 */
template <class T, class Op, bool CAsRowVector, bool CAsColVector>
void hl_gpu_apply_ternary_op(Op op,
                             T* A_d,
                             T* B_d,
                             T* C_d,
                             int dimM,
                             int dimN,
                             int lda,
                             int ldb,
                             int ldc) {
  CHECK_NOTNULL(A_d);

  if ((CAsRowVector == 0 && CAsColVector == 0) &&
      ((dimM == 1) || (dimN == lda && dimN == ldb && dimN == ldc))) {
    int size = dimM * dimN;
    int blockSize = size <= 1024 ? size : 1024;
    int gridSize = (size + 1024 - 1) / 1024;
    KeEltWiseTernaryOp<T, Op><<<gridSize, blockSize, 0, STREAM_DEFAULT>>>
      (A_d, B_d, C_d, size, op);
  } else {
    int blockSizeY = std::min(32, dimM);
    int blockSizeX = (32 / blockSizeY) * 32;
    int gridSizeX = std::min(32, (dimN + blockSizeX - 1) / blockSizeX);
    int gridSizeY = std::min(32, (dimM + blockSizeY - 1) / blockSizeY);
    dim3 threads(blockSizeX, blockSizeY);
    dim3 grid(gridSizeX, gridSizeY);
    KeEltWiseTernaryOp<T, Op, CAsRowVector, CAsColVector>
      <<<grid, threads, 0, STREAM_DEFAULT>>>
      (A_d, B_d, C_d, dimM, dimN, lda, ldb, ldc, op);
  }

  CHECK_SYNC("hl_gpu_apply_ternary_op failed");
}


/**
 * @brief   gpu element wise quaternary operator.
 */
template <class T, class Op>
void hl_gpu_apply_quaternary_op(Op op,
                                T* A_d,
                                T* B_d,
                                T* C_d,
                                T* D_d,
                                int dimM,
                                int dimN,
                                int lda,
                                int ldb,
                                int ldc,
                                int ldd) {
  CHECK_NOTNULL(A_d);

  if ((dimM == 1) ||
      (dimN == lda && dimN == ldb && dimN == ldc && dimN == ldd)) {
    int size = dimM * dimN;
    int blockSize = size <= 1024 ? size : 1024;
    int gridSize = (size + 1024 - 1) / 1024;
    KeEltWiseQuaternaryOp<T, Op><<<gridSize, blockSize, 0, STREAM_DEFAULT>>>
      (A_d, B_d, C_d, D_d, size, op);
  } else {
    int blockSizeY = std::min(32, dimM);
    int blockSizeX = (32 / blockSizeY) * 32;
    int gridSizeX = std::min(32, (dimN + blockSizeX - 1) / blockSizeX);
    int gridSizeY = std::min(32, (dimM + blockSizeY - 1) / blockSizeY);
    dim3 threads(blockSizeX, blockSizeY);
    dim3 grid(gridSizeX, gridSizeY);
    KeEltWiseQuaternaryOp<T, Op><<<grid, threads, 0, STREAM_DEFAULT>>>
      (A_d, B_d, C_d, D_d, dimM, dimN, lda, ldb, ldc, ldd, op);
  }

  CHECK_SYNC("hl_gpu_apply_quaternary_op failed");
}

#else

template <class T, class Op>
void hl_gpu_apply_unary_op(Op op, T* A_d, int dimM, int dimN, int lda) {}

template <class T, class Op, bool BAsRowVector, bool BAsColVector>
void hl_gpu_apply_binary_op(Op op,
                            T* A_d,
                            T* B_d,
                            int dimM,
                            int dimN,
                            int lda,
                            int ldb) {}

template <class T, class Op, bool CAsRowVector, bool CAsColVector>
void hl_gpu_apply_ternary_op(Op op,
                             T* A_d,
                             T* B_d,
                             T* C_d,
                             int dimM,
                             int dimN,
                             int lda,
                             int ldb,
                             int ldc) {}

template <class T, class Op>
void hl_gpu_apply_quaternary_op(Op op,
                                T* A_d,
                                T* B_d,
                                T* C_d,
                                T* D_d,
                                int dimM,
                                int dimN,
                                int lda,
                                int ldb,
                                int ldc,
                                int ldd) {}
#endif

#ifdef __NVCC__
/**
 * @brief   matrix row operator.
 */

template<class Agg, class Op>
__device__ __inline__ real sumRow(Agg agg, Op op,
                                  int idx, int blockSize,
                                  int dimN, real *A) {
  real tmp = agg.init();
  int cnt = (dimN + blockSize -1) / blockSize;
  for (int i = 0; i < cnt && idx < dimN; i++) {
      tmp = agg(tmp, op(A[idx]));
      idx += blockSize;
  }
  return tmp;
}

template<class Agg, class Op>
__device__ __inline__ real sumRow(Agg agg, Op op,
                                  int idx, int blockSize,
                                  int dimN, real *A, real *B) {
  real tmp = agg.init();
  int cnt = (dimN + blockSize -1) / blockSize;
  for (int i = 0; i < cnt && idx < dimN; i++) {
    tmp = agg(tmp, op(A[idx], B[idx]));
    idx += blockSize;
  }
  return tmp;
}

template<class Agg>
__device__ __inline__ void aggRow(Agg agg, real *row, int size, int tid) {
  for (int stride = size/2; stride > 0; stride = stride/2) {
    if (tid < stride) {
      row[tid] = agg(row[tid], row[tid + stride]);
    }
    __syncthreads();
  }
}

template<class Agg, class Op, class Saver, int blockSize>
__global__ void KeMatrixRowOp(Agg agg, Op op, Saver sv,
                              int dimN,
                              real *dst, int ld,
                              real *A, int lda) {
  __shared__ real row_s[blockSize];
  int rowId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x;

  A += rowId*lda;
  row_s[tid] = sumRow(agg, op, tid, blockSize, dimN, A);
  __syncthreads();

  aggRow(agg, row_s, blockSize, tid);
  __syncthreads();

  if (tid == 0) {
    dst[rowId*ld] = sv(dst[rowId*ld], row_s[0]);
  }
}

template<class Agg, class Op, class Saver, int blockSize>
__global__ void KeMatrixRowOp(Agg agg, Op op, Saver sv,
                              int dimN,
                              real *dst, int ld,
                              real *A, int lda,
                              real *B, int ldb) {
  __shared__ real row_s[blockSize];
  int rowId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x;

  A += rowId*lda;
  B += rowId*ldb;
  row_s[tid] = sumRow(agg, op, tid, blockSize, dimN, A, B);
  __syncthreads();

  aggRow(agg, row_s, blockSize, tid);
  __syncthreads();

  if (tid == 0) {
    dst[rowId*ld] = sv(dst[rowId*ld], row_s[0]);
  }
}

/**
 * @brief   matrix column operator.
 */
template <class Agg, class Op>
__device__ __inline__ real sumCol(Agg agg, Op op,
                                  int index, int stride,
                                  int dimM, real *A, int lda) {
  real tmp = agg.init();
  for (; index < dimM;) {
    tmp = agg(tmp, op(A[index*lda]));
    index += stride;
  }
  return tmp;
}

template <class Agg, class Op>
__device__ __inline__ real sumCol(Agg agg, Op op,
                                  int index, int stride, int dimM,
                                  real *A, int lda, real *B, int ldb) {
  real tmp = agg.init();
  for (; index < dimM;) {
    tmp = agg(tmp, op(A[index*lda], B[index*ldb]));
    index += stride;
  }
  return tmp;
}

template <class Agg, class Op, class Saver>
__global__ void KeMatrixColumnOp(Agg agg, Op op, Saver sv,
                                 int dimM, int dimN,
                                 real *dst,
                                 real *A, int lda) {
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rowIdx < dimN) {
    A += rowIdx;
    real tmp = sumCol(agg, op, 0, 1, dimM, A, lda);
    dst[rowIdx] = sv(dst[rowIdx], tmp);
  }
}

template <class Agg, class Op, class Saver, int blockDimX, int blockDimY>
__global__ void KeMatrixColumnOp_S(Agg agg, Op op, Saver sv,
                                   int dimM, int dimN,
                                   real *dst,
                                   real *A, int lda) {
  __shared__ real col_s[blockDimX*blockDimY];
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (rowIdx < dimN) {
    A += rowIdx;
    real tmp = sumCol(agg, op, threadIdx.y, blockDimY, dimM, A, lda);
    col_s[threadIdx.x + threadIdx.y*blockDimX] = tmp;
  }
  __syncthreads();

  if (rowIdx < dimN) {
    if (threadIdx.y ==0) {
      real tmp = agg.init();
      for (int i=0; i < blockDimY; i++) {
        tmp = agg(tmp, col_s[threadIdx.x + i*blockDimX]);
      }
      dst[rowIdx] = sv(dst[rowIdx], tmp);
    }
  }
}

template <class Agg, class Op, class Saver>
__global__ void KeMatrixColumnOp(Agg agg, Op op, Saver sv,
                                 int dimM, int dimN,
                                 real *dst,
                                 real *A, int lda,
                                 real *B, int ldb) {
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (rowIdx < dimN) {
    A += rowIdx;
    B += rowIdx;
    real tmp = sumCol(agg, op, 0, 1, dimM, A, lda, B, ldb);
    dst[rowIdx] = sv(dst[rowIdx], tmp);
  }
}

template <class Agg, class Op, class Saver, int blockDimX, int blockDimY>
__global__ void KeMatrixColumnOp_S(Agg agg, Op op, Saver sv,
                                   int dimM, int dimN,
                                   real *dst,
                                   real *A, int lda,
                                   real *B, int ldb) {
  __shared__ real col_s[blockDimX*blockDimY];
  int rowIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (rowIdx < dimN) {
    A += rowIdx;
    B += rowIdx;
    real tmp = sumCol(agg, op,
        threadIdx.y, blockDimY, dimM, A, lda, B, ldb);
    col_s[threadIdx.x + threadIdx.y*blockDimX] = tmp;
  }
  __syncthreads();

  if (rowIdx < dimN) {
    if (threadIdx.y ==0) {
      real tmp = agg.init();
      for (int i=0; i < blockDimY; i++) {
        tmp = agg(tmp, col_s[threadIdx.x + i*blockDimX]);
      }
      dst[rowIdx] = sv(dst[rowIdx], tmp);
    }
  }
}

#endif

template <class Agg, class Op, class Saver>
void hl_gpu_matrix_row_op(Agg agg, Op op, Saver sv,
                          int dimM, int dimN,
                          real *dst, int ld,
                          real *A, int lda) {
#ifdef __NVCC__
  CHECK_NOTNULL(dst);
  CHECK_NOTNULL(A);

  int blocksX = dimM;
  int blocksY = 1;
  dim3 threads(128, 1);
  dim3 grid(blocksX, blocksY);
  KeMatrixRowOp<Agg, Op, Saver, 128><<< grid, threads, 0, STREAM_DEFAULT >>>
      (agg, op, sv, dimN, dst, ld, A, lda);

  CHECK_SYNC("hl_matrix_row_op failed");
#endif
}

template <class Agg, class Op, class Saver>
void hl_gpu_matrix_row_op(Agg agg, Op op, Saver sv,
                          int dimM, int dimN,
                          real *dst, int ld,
                          real *A, int lda,
                          real *B, int ldb) {
#ifdef __NVCC__
  CHECK_NOTNULL(dst);
  CHECK_NOTNULL(A);

  int blocksX = dimM;
  int blocksY = 1;
  dim3 threads(128, 1);
  dim3 grid(blocksX, blocksY);
  KeMatrixRowOp<Agg, Op, Saver, 128><<< grid, threads, 0, STREAM_DEFAULT >>>
    (agg, op, sv, dimN, dst, ld, A, lda, B, ldb);

  CHECK_SYNC("hl_matrix_row_op failed");
#endif
}

template <class Agg, class Op, class Saver>
void hl_gpu_matrix_column_op(Agg agg, Op op, Saver sv,
                             int dimM, int dimN,
                             real *dst,
                             real *A, int lda) {
#ifdef __NVCC__
  if (dimN >= 8192) {
    int blocksX = (dimN + 128 -1) / 128;
    int blocksY = 1;
    dim3 threads(128, 1);
    dim3 grid(blocksX, blocksY);
    KeMatrixColumnOp<Agg, Op, Saver>
        <<< grid, threads, 0, STREAM_DEFAULT >>>
        (agg, op, sv, dimM, dimN, dst, A, lda);
  } else {
    int blocksX = (dimN + 32 -1) / 32;
    int blocksY = 1;
    dim3 threads(32, 32);
    dim3 grid(blocksX, blocksY);
    KeMatrixColumnOp_S<Agg, Op, Saver, 32, 32>
        <<< grid, threads, 0, STREAM_DEFAULT>>>
        (agg, op, sv, dimM, dimN, dst, A, lda);
  }

  CHECK_SYNC("hl_matrix_column_op failed");
#endif
}

template <class Agg, class Op, class Saver>
void hl_gpu_matrix_column_op(Agg agg, Op op, Saver sv,
                             int dimM, int dimN,
                             real *dst,
                             real *A, int lda,
                             real *B, int ldb) {
#ifdef __NVCC__
  if (dimN >= 8192) {
    int blocksX = (dimN + 128 -1) / 128;
    int blocksY = 1;
    dim3 threads(128, 1);
    dim3 grid(blocksX, blocksY);
    KeMatrixColumnOp<Agg, Op, Saver>
        <<< grid, threads, 0, STREAM_DEFAULT >>>
        (agg, op, sv, dimM, dimN, dst, A, lda, B, ldb);
  } else {
    int blocksX = (dimN + 32 -1) / 32;
    int blocksY = 1;
    dim3 threads(32, 32);
    dim3 grid(blocksX, blocksY);
    KeMatrixColumnOp_S<Agg, Op, Saver, 32, 32>
        <<< grid, threads, 0, STREAM_DEFAULT>>>
        (agg, op, sv, dimM, dimN, dst, A, lda, B, ldb);
  }

  CHECK_SYNC("hl_matrix_column_op failed");
#endif
}

#endif /* HL_GPU_MATRIX_KERNEL_CUH_ */
