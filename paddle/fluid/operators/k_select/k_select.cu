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

#include "k_select.h"

#define FABS(a) ((a != -INFINITY) ? fabs(a) : a)
#include "dense2csr.h"

__device__ int blockSyncCnt = 0;
__device__ int blockSyncCnt2 = 0;

unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor) {
  return ((dividend % divisor) == 0) ? (dividend / divisor)
                                     : (dividend / divisor + 1);
}

void getNumBlocksAndThreads(int n, int& blocks, int& threads) {
  if (n == 1) {
    threads = 1;
    blocks = 1;
  } else {
    threads = (n < MAX_THREADS) ? nextPow2(n / 2) : MAX_THREADS;
    blocks = max(1, n / (threads * 2));
  }
  blocks = min(MAX_BLOCKS, blocks);
}

int get_buffer_size(int count) {
  int blocks;
  int threads;
  getNumBlocksAndThreads(count, blocks, threads);
  return 2 * blocks * threads;
}

template <typename T>
struct Pair {
  __device__ __forceinline__ Pair() {}
  __device__ __forceinline__ Pair(T value, int id) : v(value), id(id) {}

  __device__ __forceinline__ void set(T value, int id) {
    v = value;
    id = id;
  }

  __device__ __forceinline__ void operator=(const Pair<T>& in) {
    v = in.v;
    id = in.id;
  }

  __device__ __forceinline__ bool operator<(const T value) const {
    return (FABS(v) < FABS(value));
  }

  __device__ __forceinline__ bool operator<(const Pair<T>& in) const {
    return (FABS(v) < FABS(in.v)) || ((FABS(v) == FABS(in.v)) && (id > in.id));
  }

  __device__ __forceinline__ bool operator>(const Pair<T>& in) const {
    return (FABS(v) > FABS(in.v)) || ((FABS(v) == FABS(in.v)) && (id < in.id));
  }

  T v;
  int id;
};

template <typename T>
__device__ __forceinline__ void AddTo(Pair<T> topk[], const Pair<T>& p,
                                      int beam_size) {
  for (int k = beam_size - 2; k >= 0; k--) {
    if (topk[k] < p) {
      topk[k + 1] = topk[k];
    } else {
      topk[k + 1] = p;
      return;
    }
  }
  topk[0] = p;
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      AddTo<T>(topk, tmp, beam_size);
    }
    idx += BlockSize;
  }
}

template <typename T, int BlockSize>
__device__ __forceinline__ void GetTopK(Pair<T> topk[], const T* src, int idx,
                                        int dim, const Pair<T>& max,
                                        int beam_size) {
  while (idx < dim) {
    if (topk[beam_size - 1] < src[idx]) {
      Pair<T> tmp(src[idx], idx);
      if (tmp < max) {
        AddTo<T>(topk, tmp, beam_size);
      }
    }
    idx += BlockSize;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ void ThreadGetTopK(Pair<T> topk[], int* beam,
                                              int beam_size, const T* src,
                                              bool* firstStep, bool* is_empty,
                                              Pair<T>* max, int dim,
                                              const int tid) {
  if (*beam > 0) {
    int length = (*beam) < beam_size ? *beam : beam_size;
    if (*firstStep) {
      *firstStep = false;
      GetTopK<T, BlockSize>(topk, src, tid, dim, length);
    } else {
      for (int k = 0; k < MaxLength; k++) {
        if (k < MaxLength - (*beam)) {
          topk[k] = topk[k + *beam];
        } else {
          topk[k].set(-INFINITY, -1);
        }
      }
      if (!(*is_empty)) {
        GetTopK<T, BlockSize>(topk + MaxLength - *beam, src, tid, dim, *max,
                              length);
      }
    }

    *max = topk[MaxLength - 1];
    if ((*max).v == -1) *is_empty = true;
    *beam = 0;
  }
}

template <typename T, int MaxLength, int BlockSize>
__device__ __forceinline__ T BlockReduce(Pair<T>* sh_topk, int* maxid,
                                         Pair<T> topk[], int* beam, int* k,
                                         const int tid, const int warp,
                                         T** topVal = nullptr,
                                         int** topIds = nullptr) {
  T ret = -INFINITY;
  while (true) {
    __syncthreads();
    if (tid < BlockSize / 2) {
      if (sh_topk[tid] < sh_topk[tid + BlockSize / 2]) {
        maxid[tid] = tid + BlockSize / 2;
      } else {
        maxid[tid] = tid;
      }
    }
    __syncthreads();
    for (int stride = BlockSize / 4; stride > 0; stride = stride / 2) {
      if (tid < stride) {
        if (sh_topk[maxid[tid]] < sh_topk[maxid[tid + stride]]) {
          maxid[tid] = maxid[tid + stride];
        }
      }
      __syncthreads();
    }
    __syncthreads();

    if (tid == 0 && topVal != nullptr && topIds != nullptr) {
      **topVal = sh_topk[maxid[0]].v;
      **topIds = sh_topk[maxid[0]].id;
      (*topVal)++;
      (*topIds)++;
    }
    if (tid == maxid[0]) (*beam)++;
    if (--(*k) == 0) {
      if (tid == 0) {
        ret = sh_topk[maxid[0]].v;
      }
      break;
    }
    __syncthreads();

    if (tid == maxid[0]) {
      if (*beam < MaxLength) {
        sh_topk[tid] = topk[*beam];
      }
    }

    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, true);

    if (maxid[0] / 32 == warp) {
      if (CudaShuffleSync(mask, *beam, (maxid[0]) % 32, 32) == MaxLength) break;
    }
  }
  return ret;
}

template <typename T, int MaxLength, int BlockSize>
__global__ void KeGetTopKShort(T* output, const T* src, int lds, int dim, int k,
                               T* value, int* index) {
  __shared__ Pair<T> sh_topk[BlockSize];
  __shared__ int maxid[BlockSize / 2];
  const int warp = threadIdx.x / 32;
  T kth = -INFINITY;

  Pair<T> topk[MaxLength];
  int beam = MaxLength;
  Pair<T> max;
  bool is_empty = false;
  bool firststep = true;

  for (int k = 0; k < MaxLength; k++) {
    topk[k].set(-INFINITY, -1);
  }
  while (k) {
    ThreadGetTopK<T, MaxLength, BlockSize>(topk, &beam, k,
                                           src + blockIdx.x * lds, &firststep,
                                           &is_empty, &max, dim, threadIdx.x);

    sh_topk[threadIdx.x] = topk[0];
    T temp = BlockReduce<T, MaxLength, BlockSize>(
        sh_topk, maxid, topk, &beam, &k, threadIdx.x, warp, &value, &index);
    if (temp != -INFINITY) {
      kth = temp;
    }
  }
  if (kth != -INFINITY) {
    output[blockIdx.x] = kth;
  }
}

template <typename T, int MaxLength, int BlockSize>
__global__ void KeGetSampleTopK(T* output, const T* src, int lds, int dim,
                                int k) {
  __shared__ Pair<T> sh_topk[BlockSize];
  __shared__ int maxid[BlockSize / 2];
  const int warp = threadIdx.x / 32;
  T kth = -INFINITY;

  Pair<T> topk[MaxLength];
  int beam = MaxLength;
  Pair<T> max;
  bool is_empty = false;
  bool firststep = true;

  for (int k = 0; k < MaxLength; k++) {
    topk[k].set(-INFINITY, -1);
  }
  while (k) {
    ThreadGetTopK<T, MaxLength, BlockSize>(topk, &beam, k,
                                           src + blockIdx.x * lds, &firststep,
                                           &is_empty, &max, dim, threadIdx.x);

    sh_topk[threadIdx.x] = topk[0];
    T temp = BlockReduce<T, MaxLength, BlockSize>(sh_topk, maxid, topk, &beam,
                                                  &k, threadIdx.x, warp);
    if (temp != -INFINITY) {
      kth = temp;
    }
  }
  if (kth != -INFINITY) {
    output[blockIdx.x] = kth;
  }
}

template <typename T, int BlockSize>
__global__ void KeGetTotalTopk(volatile T* data, int n) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  T res = 0;
  if (bid == 0 && tid == 0) {
    for (int i = 0; i < gridDim.x; i++) {
      res += FABS(data[i]);
    }
  }
  __syncthreads();
  if (bid == 0 && tid == 0) {
    data[0] = res / n;
  }
}

template <typename T>
__device__ void findMinMaxCrossBlock(T thr_min, T thr_max, T* sh_min, T* sh_max,
                                     int bid, int blocksize, T* minmax) {
  int warp_id = threadIdx.x / warpSize;
  int lane_id = threadIdx.x % warpSize;
  int warp_nb = (blocksize % warpSize == 0) ? (blocksize / warpSize)
                                            : (blocksize / warpSize + 1);

#pragma unroll
  for (int i = 1; i <= warpSize; i *= 2) {
    uint32_t m = 0xffffffff;
    T min_val = CudaShuffleUpSync(m, thr_min, i, warpSize);
    T max_val = CudaShuffleUpSync(m, thr_max, i, warpSize);
    if (lane_id >= i) {
      if (min_val < thr_min) thr_min = min_val;
      if (max_val > thr_max) thr_max = max_val;
    }
  }
  if (lane_id == warpSize - 1) {
    sh_min[warp_id] = thr_min;
    sh_max[warp_id] = thr_max;
  }
  __syncthreads();

  if (warp_id == 0 && lane_id < warp_nb) {
    T warp_min = sh_min[lane_id];
    T warp_max = sh_max[lane_id];
    int m = (1 << warp_nb) - 1;
    for (int i = 1; i <= warp_nb; i *= 2) {
      T min_val = CudaShuffleUpSync(m, warp_min, i, warp_nb);
      T max_val = CudaShuffleUpSync(m, warp_max, i, warp_nb);
      if (lane_id >= i) {
        if (min_val < warp_min) warp_min = min_val;
        if (max_val > warp_max) warp_max = max_val;
      }
    }
    sh_min[lane_id] = warp_min;
    sh_max[lane_id] = warp_max;
  }
  __syncthreads();

  if (lane_id == warpSize - 1 && minmax != NULL) {
    thr_min = sh_min[warp_nb - 1];
    thr_max = sh_max[warp_nb - 1];
    minmax[bid * 2] = thr_max;
    minmax[bid * 2 + 1] = thr_min;
    __threadfence();
  }
}

template <typename T>
__device__ void findMinMaxCrossBlock(T thr_min, T thr_max, T* sh_min, T* sh_max,
                                     T* minmax = NULL) {
  findMinMaxCrossBlock(thr_min, thr_max, sh_min, sh_max, blockIdx.x, blockDim.x,
                       minmax);
}

template <typename T>
__global__ void KeMinMax(const T* input, int count, T* minmax, int chunks) {
  extern __shared__ T shared_memory[];
  __shared__ int islast;
  T* sh_max = shared_memory;
  T* sh_min = shared_memory + blockDim.x;
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  T thr_min = FABS(input[id]);
  T thr_max = thr_min;
  id += blockDim.x * gridDim.x;
  for (int i = id; i < count; i += blockDim.x * gridDim.x) {
    float val = FABS(input[i]);
    if (val < thr_min) {
      thr_min = val;
    } else if (val > thr_max) {
      thr_max = val;
    }
  }

  findMinMaxCrossBlock<T>(thr_min, thr_max, sh_min, sh_max, minmax + 2);
  __syncthreads();

  if (threadIdx.x == 0) {
    int value = atomicAdd(&blockSyncCnt, 1);
    islast = (value == gridDim.x - 1);
  }
  __syncthreads();

  if (islast) {
    blockSyncCnt = 0;
    T min_val = INFINITY;
    T max_val = -INFINITY;
    if (threadIdx.x < gridDim.x) {
      max_val = minmax[2 * threadIdx.x + 2];
      min_val = minmax[2 * threadIdx.x + 3];
    }
    findMinMaxCrossBlock<T>(min_val, max_val, sh_min, sh_max, 0, gridDim.x,
                            minmax);
    __syncthreads();
    if (threadIdx.x == 0) {
      max_val = minmax[0];
      min_val = minmax[1];
      minmax[1] = (max_val - min_val) / chunks;
    }
  }
}

template <typename T, int MAX_LENGTH>
__device__ void bucketCountCrossBlock(int* cnt, int* sh_cnt, int* block_count) {
  int warp_id = threadIdx.x / warpSize;
  int lane_id = threadIdx.x % warpSize;
  int warp_nb = blockDim.x / warpSize;
  for (int j = 0; j < MAX_LENGTH; j++) {
#pragma unroll
    for (int i = 1; i <= warpSize; i *= 2) {
      uint32_t m = 0xffffffff;
      int n = CudaShuffleUpSync(m, cnt[j], i, warpSize);
      if (lane_id >= i) cnt[j] += n;
    }
    if (lane_id == warpSize - 1) {
      sh_cnt[warp_id * MAX_LENGTH + j] = cnt[j];
    }
  }
  __syncthreads();

  for (int j = 0; j < MAX_LENGTH; j++) {
    if (warp_id == 0 && lane_id < warp_nb) {
      int warp_sum = sh_cnt[j + lane_id * MAX_LENGTH];
      int m = (1 << warp_nb) - 1;
      for (int i = 1; i <= warp_nb; i *= 2) {
        int n = CudaShuffleUpSync(m, warp_sum, i, warp_nb);
        if (lane_id >= i) warp_sum += n;
      }
      sh_cnt[j + lane_id * MAX_LENGTH] = warp_sum;
    }
  }
  __syncthreads();

  if (lane_id == warpSize - 1) {
    for (int j = 0; j < MAX_LENGTH; j++) {
      block_count[blockIdx.x * MAX_LENGTH + j] =
          sh_cnt[(warp_nb - 1) * MAX_LENGTH + j];
    }
    __threadfence();
  }
}

template <typename T, int MAX_LENGTH>
__device__ void bucketCount(const T* input, int count, int* block_count,
                            T* pmax, T* pdel, int* pk, int* pkmin) {
  extern __shared__ int shm[];
  __shared__ int islast;
  int* sh_cnt = shm;
  T max = *pmax;
  T del = *pdel;
  if (del <= 0) return;
  int cnt[MAX_LENGTH];

  for (int i = 0; i < MAX_LENGTH; i++) cnt[i] = 0;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    T val = FABS(input[i]);
    if (val > max) continue;
    int index = (max - val) / del;
    if (index < MAX_LENGTH) {
      cnt[index]++;
    }
  }

  bucketCountCrossBlock<T, MAX_LENGTH>(cnt, sh_cnt, block_count);

  if (threadIdx.x == 0) {
    int value = atomicAdd(&blockSyncCnt, 1);
    islast = (value == gridDim.x - 1);
  }
  __syncthreads();

  if (islast && threadIdx.x == 0) {
    blockSyncCnt = 0;
    for (int i = 1; i < gridDim.x; i++) {
      for (int j = 0; j < MAX_LENGTH; j++) {
        block_count[j] += block_count[j + i * MAX_LENGTH];
      }
    }
    for (int j = 1; j < MAX_LENGTH; j++) {
      block_count[j] += block_count[j - 1];
    }
    int k = *pk;
    int kmin = *pkmin;
    int prev_cnt = 0;
    int j = 0;
    for (; j < MAX_LENGTH; j++) {
      if (block_count[j] > k) {
        if (j == 0) {
          *pdel = del / 4;
          break;
        }
        prev_cnt = block_count[j - 1];
        if (prev_cnt < kmin) {
          *pdel = del / 4;
          *pmax = max - j * del;
          *pk -= prev_cnt;
          *pkmin -= prev_cnt;
          break;
        }
        *pmax = max - j * del;
        *pdel = 0;
        break;
      } else if (block_count[j] == k) {
        *pdel = 0;
        *pmax = max - (j + 1) * del;
        *pdel = 0;
        break;
      }
    }
    if (j == MAX_LENGTH) {
      prev_cnt = block_count[MAX_LENGTH - 1];
      *pmax = max - MAX_LENGTH * del;
      *pk -= prev_cnt;
      *pkmin -= prev_cnt;
    }
    __threadfence();
    return;
  }
}

template <typename T, int MAX_LENGTH>
__global__ void KeFindKth(const T* input, int count, int* block_count, T* pmax,
                          T* pdel, int* pk, int* pkmin, int first, int k = -1) {
  if (first == 1) {
    *pk = k;
    *pkmin = (int)((float)k * 0.8);
  } else {
    if (*pdel == 0) return;
  }
  bucketCount<T, MAX_LENGTH>(input, count, block_count, pmax, pdel, pk, pkmin);
}

template <typename T>
__global__ void KeGetThreadCountByThreshold(const T* idata, int* odata,
                                            int count, T* threshold) {
  extern int __shared__ sdata[];
  sdata[threadIdx.x] = 0;
  __syncthreads();
  T kth = *threshold;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    if (FABS(idata[i]) >= kth) {
      sdata[threadIdx.x]++;
    }
  }
  __syncthreads();
  odata[threadIdx.x + blockDim.x * blockIdx.x] = sdata[threadIdx.x];
}

__global__ void KePrefixSum(int* data, int width, int* partial_sums = NULL) {
  extern __shared__ int shm[];
  int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
  int lane_id = id % warpSize;
  int warp_id = threadIdx.x / warpSize;
  int value = data[id];

#pragma unroll
  for (int i = 1; i <= width; i *= 2) {
    unsigned int mask = 0xffffffff;
    int n = CudaShuffleUpSync(mask, value, i, width);
    if (lane_id >= i) value += n;
  }

  if (threadIdx.x % warpSize == warpSize - 1) {
    shm[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
    int warp_sum = shm[lane_id];
    int mask = (1 << (blockDim.x / warpSize)) - 1;
    for (int i = 1; i <= (blockDim.x / warpSize); i *= 2) {
      int n = CudaShuffleUpSync(mask, warp_sum, i, (blockDim.x / warpSize));
      if (lane_id >= i) warp_sum += n;
    }
    shm[lane_id] = warp_sum;
  }
  __syncthreads();

  int blockSum = 0;
  if (warp_id > 0) {
    blockSum = shm[warp_id - 1];
  }
  value += blockSum;
  data[id] = value;

  if (partial_sums != NULL && threadIdx.x == blockDim.x - 1) {
    partial_sums[blockIdx.x] = value;
  }
}

__global__ void KeGlobalPrefixSum(int* data, int* partial_sums, int len,
                                  int k) {
  __shared__ int buf;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id > len) return;
  if (threadIdx.x == 0) {
    buf = partial_sums[blockIdx.x];
  }
  __syncthreads();
  data[id] += buf;
}

template <typename T>
__global__ void KeEncode(const T* data, int count, int* scan, T* value,
                         int* index, T* threshold, int k) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int start = 0;
  if (id == 0) {
    start = 0;
  } else {
    start = scan[id - 1];
  }
  __syncthreads();

  T kth = *threshold;
  int offset = start;

  for (int i = id; i < count; i += gridDim.x * blockDim.x) {
    T val = data[i];
    if (offset < k && FABS(val) > kth) {
      index[offset] = i;
      value[offset] = val;
      offset++;
    }
  }
}

template <typename T>
__global__ void KeMask(int* index, int k, T* data) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = id; i < k; i += gridDim.x * blockDim.x) {
    int idx = index[i];
    data[idx] = 0;
  }
}

void get_threshold(float* threshold, float* input, int count, int k,
                   cudaStream_t stream) {
  const int max_length = 5;
  int sampk = min(32, k / 2);
  const float sample_prop = static_cast<float>(k) / sampk;
  int sampdim = count / sample_prop;

  const int BlockSize = 1024;
  int threads = BlockSize;
  int blocks = min(16, static_cast<int>(sample_prop / 2));
  int lds = count / blocks;

  KeGetSampleTopK<float, max_length, BlockSize><<<blocks, threads, 0, stream>>>(
      threshold, input, lds, sampdim, sampk);
  KeGetTotalTopk<float, BlockSize><<<blocks, threads, 0, stream>>>(threshold,
                                                                   blocks);
}

bool k_select_min(float* input, int count, void* encode, void* buff, int k,
                  cudaStream_t stream, float* moment) {
  if (count < MIN_COUNT_FOR_SHORT_TOPK) {
    return false;
  }
  if (count < MIN_COUNT_FOR_LARGE_TOPK) {
    const int MaxLength = 60;
    const int BlockSize = 512;
    int blocks = 1;
    int threads = BlockSize;
    float* value = static_cast<float*>(encode) + k;
    int* index = static_cast<int*>(encode);
    float* threshold = static_cast<float*>(buff);
    KeGetTopKShort<float, MaxLength, BlockSize><<<blocks, threads, 0, stream>>>(
        threshold, input, count, count, k, value, index);
    KeMask<float><<<GET_BLOCKS(k), CUDA_NUM_THREADS, 0, stream>>>(index, k,
                                                                  input);
    if (moment != nullptr) {
      KeMask<float><<<GET_BLOCKS(k), CUDA_NUM_THREADS, 0, stream>>>(index, k,
                                                                    moment);
    }
    return true;
  }
  return false;
}

void dense2coo(void* encode, float* input, float* threshold, int* thr_cnt,
               int count, int k, cudaStream_t stream, float* moment) {
  int blocks, threads;
  getNumBlocksAndThreads(count, blocks, threads);
  int smemSize = sizeof(float) * threads;
  int p_threads = min(blocks, threads);
  int p_blocks = iDivUp(blocks, p_threads);

  KeGetThreadCountByThreshold<float><<<blocks, threads, smemSize, stream>>>(
      input, thr_cnt, count, threshold);

  /*
  int* part = thr_cnt + threads * blocks;
  KePrefixSum<<<blocks, threads, smemSize, stream>>>(thr_cnt, 32, part);
  KePrefixSum<<<p_blocks, p_threads, smemSize, stream>>>(part, 32);
  KeGlobalPrefixSum<<<blocks - 1, threads,0, stream>>>(thr_cnt + threads, part, count, k);
  int* index = static_cast<int*>(encode);
  float* value = static_cast<float*>(encode) + k;
  KeEncode<float><<<blocks, threads, 0, stream>>>(input, count, thr_cnt, value,
                                                  index, threshold, k);
  KeMask<float><<<GET_BLOCKS(k), CUDA_NUM_THREADS, 0, stream>>>(index, k,
                                                                input);
  if (moment != nullptr) {
    KeMask<float><<<GET_BLOCKS(k), CUDA_NUM_THREADS, 0, stream>>>(index, k,
                                                                  moment);
  }
  */
}

void get_threshold_bucket(void* buff, float* input, int count, int k,
                          cudaStream_t stream) {
  int blocks;
  int threads;
  getNumBlocksAndThreads(count, blocks, threads);
  int smemSize = sizeof(float) * threads * 2;
  int chunks = ((float)count / k) * (BUCKETS / 4);
  KeMinMax<float><<<blocks, threads, smemSize, stream>>>(input, count,
                                                         (float*)buff, chunks);
  smemSize = sizeof(int) * BUCKETS * threads * 2;
  float* threshold = static_cast<float*>(buff);
  float* pdel = static_cast<float*>(buff) + 1;
  int* pk = static_cast<int*>(buff) + 2;
  int* pkmin = static_cast<int*>(buff) + 3;
  int* block_count = static_cast<int*>(buff) + 4;

  KeFindKth<float, BUCKETS><<<blocks, threads, smemSize, stream>>>(
      input, count, block_count, threshold, pdel, pk, pkmin, 1, k);
  KeFindKth<float, BUCKETS><<<blocks, threads, smemSize, stream>>>(
      input, count, block_count, threshold, pdel, pk, pkmin, 0);
}

bool k_select_bucket(float* input, int count, void* encode, void* buff, int k,
                     int protocal, cudaStream_t stream, float* moment) {
  get_threshold_bucket(buff, input, count, k, stream);

  float* threshold = static_cast<float*>(buff);
  int* thr_cnt = static_cast<int*>(buff) + 4;

  if (protocal == 0) {
    dense2coo(encode, input, threshold, thr_cnt, count, k, stream, moment);
  } else if (protocal == 1) {
    dense2csr(encode, input, threshold, thr_cnt, count, k, stream);
  }

  return true;
}

bool k_select(float* input, int count, void* encode, void* buff, int k,
              int protocal, cudaStream_t stream, float* moment) {
  if (count < MIN_COUNT_FOR_LARGE_TOPK) {
    return k_select_min(input, count, encode, buff, k, stream, moment);
  }

  float* threshold = static_cast<float*>(buff);
  get_threshold(threshold, input, count, k, stream);

  int blocks=0;
  int threads=0;
  getNumBlocksAndThreads(count, blocks, threads);
  int* thr_cnt = static_cast<int*>(buff) + blocks;
  if (protocal == 0) {  // coo
    dense2coo(encode, input, threshold, thr_cnt, count, k, stream, moment);
  } else if (protocal == 1) {  // csr
    // dense2csr(encode, input, threshold, thr_cnt, count, k, stream);
    exit(-1);
  }
  return true;
}
