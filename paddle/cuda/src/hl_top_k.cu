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
#include "hl_top_k.h"
#include "hl_sparse.ph"
#include "paddle/utils/Logging.h"

// using namespace hppl;

struct Pair {
  __device__ __forceinline__
  Pair() {}

  __device__ __forceinline__
  Pair(real value, int id) : v_(value), id_(id) {}

  __device__ __forceinline__
  void set(real value, int id) {
    v_ = value;
    id_ = id;
  }

  __device__ __forceinline__
  void operator=(const Pair& in) {
    v_ = in.v_;
    id_ = in.id_;
  }

  __device__ __forceinline__
  bool operator<(const real value) const {
    return (v_ < value);
  }

  __device__ __forceinline__
  bool operator<(const Pair& in) const {
    return (v_ < in.v_) || ((v_ == in.v_) && (id_ > in.id_));
  }

  __device__ __forceinline__
  bool operator>(const Pair& in) const {
    return (v_ > in.v_) || ((v_ == in.v_) && (id_ < in.id_));
  }

  real v_;
  int id_;
};

__device__ __forceinline__
void addTo(Pair topK[], const Pair &p, int beamSize) {
  for (int k = beamSize - 2; k >= 0; k--) {
    if (topK[k] < p) {
      topK[k + 1] = topK[k];
    } else {
      topK[k + 1] = p;
      return;
    }
  }
  topK[0] = p;
}

template<int beamSize>
__device__ __forceinline__
void addTo(Pair topK[], const Pair &p) {
  for (int k = beamSize - 2; k >= 0; k--) {
    if (topK[k] < p) {
      topK[k + 1] = topK[k];
    } else {
      topK[k + 1] = p;
      return;
    }
  }
  topK[0] = p;
}

template<int blockSize>
__device__ __forceinline__
void getTopK(Pair topK[], real *src, int idx, int dim, int beamSize) {
  while (idx < dim) {
    if (topK[beamSize - 1] < src[idx]) {
      Pair tmp(src[idx], idx);
      addTo(topK, tmp, beamSize);
    }
    idx += blockSize;
  }
}

template<int blockSize>
__device__ __forceinline__
void getTopK(Pair topK[], real *src, int idx, int dim,
             const Pair& max, int beamSize) {
  while (idx < dim) {
    if (topK[beamSize - 1] < src[idx]) {
      Pair tmp(src[idx], idx);
      if (tmp < max) {
        addTo(topK, tmp, beamSize);
      }
    }
    idx += blockSize;
  }
}

template<int blockSize>
__device__ __forceinline__
void getTopK(Pair topK[], real *val, int *col,
             int idx, int dim, int beamSize) {
  while (idx < dim) {
    if (topK[beamSize - 1] < val[idx]) {
      Pair tmp(val[idx], col[idx]);
      addTo(topK, tmp, beamSize);
    }
    idx += blockSize;
  }
}

template<int blockSize>
__device__ __forceinline__
void getTopK(Pair topK[], real *val, int *col, int idx, int dim,
             const Pair& max, int beamSize) {
  while (idx < dim) {
    if (topK[beamSize - 1] < val[idx]) {
      Pair tmp(val[idx], col[idx]);
      if (tmp < max) {
        addTo(topK, tmp, beamSize);
      }
    }
    idx += blockSize;
  }
}

template<int maxLength, int blockSize>
__device__ __forceinline__
void threadGetTopK(Pair topK[], int& beam, int beamSize,
                   real* src,
                   bool& firstStep, bool& isEmpty, Pair& max,
                   int dim, const int tid) {
  if (beam > 0) {
    int length = beam < beamSize ? beam : beamSize;
    if (firstStep) {
      firstStep = false;
      getTopK<blockSize>(topK, src, tid, dim, length);
    } else {
      for (int k = 0; k < maxLength; k++) {
        if (k < maxLength - beam) {
          topK[k] = topK[k + beam];
        } else {
          topK[k].set(-HL_FLOAT_MAX, -1);
        }
      }
      if (!isEmpty) {
        getTopK<blockSize>(topK + maxLength - beam, src, tid, dim,
                           max, length);
      }
    }

    max = topK[maxLength - 1];
    if (max.id_ == -1) isEmpty = true;
    beam = 0;
  }
}

template<int maxLength, int blockSize>
__device__ __forceinline__
void threadGetTopK(Pair topK[], int& beam, int beamSize,
                   real* val, int* col,
                   bool& firstStep, bool& isEmpty, Pair& max,
                   int dim, const int tid) {
  if (beam > 0) {
    int length = beam < beamSize ? beam : beamSize;
    if (firstStep) {
      firstStep = false;
      getTopK<blockSize>(topK, val, col, tid, dim, length);
    } else {
      for (int k = 0; k < maxLength; k++) {
        if (k < maxLength - beam) {
          topK[k] = topK[k + beam];
        } else {
          topK[k].set(-HL_FLOAT_MAX, -1);
        }
      }
      if (!isEmpty) {
        getTopK<blockSize>(topK + maxLength - beam, val, col, tid, dim,
                           max, length);
      }
    }

    max = topK[maxLength - 1];
    if (max.id_ == -1) isEmpty = true;
    beam = 0;
  }
}

template<int maxLength, int blockSize>
__device__ __forceinline__
void blockReduce(Pair* shTopK, int* maxId, Pair topK[],
                 real** topVal, int** topIds,
                 int& beam, int& beamSize,
                 const int tid, const int warp) {
  while (true) {
    __syncthreads();
    if (tid < blockSize / 2) {
      if (shTopK[tid] < shTopK[tid + blockSize / 2]) {
        maxId[tid] = tid + blockSize / 2;
      } else {
        maxId[tid] = tid;
      }
    }
    __syncthreads();
    for (int stride = blockSize / 4; stride > 0; stride = stride/2) {
      if (tid < stride) {
        if (shTopK[maxId[tid]] < shTopK[maxId[tid + stride]]) {
          maxId[tid] = maxId[tid + stride];
        }
      }
      __syncthreads();
    }
    __syncthreads();

    if (tid == 0) {
      **topVal = shTopK[maxId[0]].v_;
      **topIds = shTopK[maxId[0]].id_;
      (*topVal)++;
      (*topIds)++;
    }
    if (tid == maxId[0]) beam++;
    if (--beamSize == 0) break;
    __syncthreads();

    if (tid == maxId[0]) {
      if (beam < maxLength) {
        shTopK[tid] = topK[beam];
      }
    }
    if (maxId[0] / 32 == warp) {
      if (__shfl(beam, (maxId[0]) % 32, 32) == maxLength) break;
    }
  }
}

/**
 * Each block compute one sample.
 * In a block:
 * 1. every thread get top maxLength value;
 * 2. merge to shTopK, block reduce and get max value;
 * 3. go to the second setp, until one thread's topK value is null;
 * 4. go to the first setp, until get the topK value.
 */
template<int maxLength, int blockSize>
__global__ void KeMatrixTopK(real* topVal, int ldv,
                             int * topIds,
                             real* src, int lds,
                             int dim,
                             int beamSize) {
  __shared__ Pair shTopK[blockSize];
  __shared__ int maxId[blockSize / 2];
  const int tid = threadIdx.x;
  const int warp = threadIdx.x / 32;
  src += blockIdx.x * lds;
  topVal += blockIdx.x * ldv;
  topIds += blockIdx.x * beamSize;

  Pair topK[maxLength]; // NOLINT
  int beam = maxLength;
  Pair max;
  bool isEmpty = false;
  bool firstStep = true;

  for (int k = 0; k < maxLength; k++) {
    topK[k].set(-HL_FLOAT_MAX, -1);
  }
  while (beamSize) {
    threadGetTopK<maxLength, blockSize>
      (topK, beam, beamSize, src, firstStep, isEmpty, max, dim, tid);

    shTopK[tid] = topK[0];
    blockReduce<maxLength, blockSize>
      (shTopK, maxId, topK, &topVal, &topIds, beam, beamSize, tid, warp);
  }
}

template<int maxLength, int blockSize>
__global__ void KeSMatrixTopK(real* topVal, int ldv,
                              int * topIds,
                              real* val,
                              int* row,
                              int* col,
                              int beamSize) {
  __shared__ Pair shTopK[blockSize];
  __shared__ int maxId[blockSize / 2];
  const int tid = threadIdx.x;
  const int warp = threadIdx.x / 32;
  topVal += blockIdx.x * ldv;
  topIds += blockIdx.x * beamSize;

  Pair topK[maxLength]; // NOLINT
  int beam = maxLength;
  Pair max;
  bool isEmpty = false;
  bool firstStep = true;

  int start = row[blockIdx.x];
  int end = row[blockIdx.x + 1];
  int dim = end - start;
  val += start;
  col += start;

  if (beamSize > dim) {
    // if the number of values to sort are less than the output size,
    // use -1 to indicate the end of valid sorted values.
    if (tid == 0) {
      topIds[dim] = -1;
    }

    beamSize = dim;
  }

  for (int k = 0; k < maxLength; k++) {
    topK[k].set(-HL_FLOAT_MAX, -1);
  }
  while (beamSize) {
    threadGetTopK<maxLength, blockSize>
      (topK, beam, beamSize, val, col, firstStep, isEmpty, max, dim, tid);

    shTopK[tid] = topK[0];
    blockReduce<maxLength, blockSize>
      (shTopK, maxId, topK, &topVal, &topIds, beam, beamSize, tid, warp);
  }
}

void hl_matrix_top_k(real* topVal, int ldv,
                     int * topIds,
                     real* src, int lds,
                     int dim,
                     int beamSize,
                     int numSamples) {
  CHECK_NOTNULL(topVal);
  CHECK_NOTNULL(topIds);
  CHECK_NOTNULL(src);

  if (beamSize > dim) beamSize = dim;

  dim3 threads(256, 1);
  dim3 grid(numSamples, 1);
  KeMatrixTopK<5, 256><<< grid, threads, 0, STREAM_DEFAULT >>>
    (topVal, ldv, topIds, src, lds, dim, beamSize);

  CHECK_SYNC("hl_matrix_top_k failed");
}

void hl_sparse_matrix_top_k(real* topVal, int ldv,
                            int * topIds,
                            hl_sparse_matrix_s src,
                            int beamSize,
                            int numSamples) {
  CHECK_NOTNULL(topVal);
  CHECK_NOTNULL(topIds);
  CHECK_NOTNULL(src);
  CHECK_EQ(src->format, HL_SPARSE_CSR)
    <<"sparse matrix format error!";

  hl_csr_matrix csr = (hl_csr_matrix)src->matrix;
  if (csr->csr_val == NULL || csr->csr_row == NULL ||
      csr->csr_col == NULL) {
    LOG(FATAL) << "parameter src is null!";
  }

  dim3 threads(256, 1);
  dim3 grid(numSamples, 1);
  KeSMatrixTopK<5, 256><<< grid, threads, 0, STREAM_DEFAULT >>>
    (topVal, ldv, topIds, csr->csr_val, csr->csr_row, csr->csr_col, beamSize);

  CHECK_SYNC("hl_sparse_matrix_top_k failed");
}

/**
 * Each block compute one sample.
 * In a block:
 * 1. every thread get top maxLength value;
 * 2. merge to shTopK, block reduce and get max value;
 * 3. go to the second setp, until one thread's topK value is null;
 * 4. go to the first setp, until get the topK value.
 */
template<int maxLength, int blockSize>
__global__ void KeMatrixTopKClassificationError(real* topVal, int ldv,
                                                int * topIds,
                                                real* src, int lds,
                                                int dim,
                                                int beamSize,
                                                int* label,
                                                real* recResult) {
  __shared__ Pair shTopK[blockSize];
  __shared__ int maxId[blockSize / 2];
  const int tid = threadIdx.x;
  const int warp = threadIdx.x / 32;
  src += blockIdx.x * lds;
  topVal += blockIdx.x * ldv;
  topIds += blockIdx.x * beamSize;

  Pair topK[maxLength]; // NOLINT
  int beam = maxLength;
  Pair max;
  bool isEmpty = false;
  bool firstStep = true;
  int topkSize = beamSize;

  for (int k = 0; k < maxLength; k++) {
    topK[k].set(-HL_FLOAT_MAX, -1);
  }

  while (beamSize) {
    threadGetTopK<maxLength, blockSize>
      (topK, beam, beamSize, src, firstStep, isEmpty, max, dim, tid);

    shTopK[tid] = topK[0];
    blockReduce<maxLength, blockSize>
      (shTopK, maxId, topK, &topVal, &topIds, beam, beamSize, tid, warp);
  }

  __syncthreads();
  if (tid == 0) {
    for (int i = 0; i < topkSize; i++) {
        if (*--topIds == label[blockIdx.x]) {
            recResult[blockIdx.x] = 0;
            break;
        }
        recResult[blockIdx.x] = 1.0f;
    }
  }
}

void hl_matrix_classification_error(real* topVal, int ldv,
                                   int* topIds,
                                   real* src, int lds,
                                   int dim,
                                   int topkSize,
                                   int numSamples,
                                   int* label,
                                   real* recResult) {
  CHECK_NOTNULL(topVal);
  CHECK_NOTNULL(topIds);
  CHECK_NOTNULL(src);

  if (topkSize > dim) topkSize = dim;

  dim3 threads(256, 1);
  dim3 grid(numSamples, 1);
  KeMatrixTopKClassificationError<5, 256>
  <<< grid, threads, 0, STREAM_DEFAULT >>>
  (topVal, ldv, topIds, src, lds, dim, topkSize, label, recResult);

  CHECK_SYNC("hl_matrix_top_k classification error failed");
}
