// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include "paddle/fluid/operators/k_select/k_select.h"

#define FABS(a) ((a != -INFINITY) ? fabs(a) : a)

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

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads,
                            int& blocks,     // NOLINT
                            int& threads) {  // NOLINT
  if (n == 1) {
    threads = 1;
    blocks = 1;
  } else {
    threads = (n < maxThreads) ? nextPow2(n / 2) : maxThreads;
    blocks = max(1, n / (threads * 2));
  }
  blocks = min(maxBlocks, blocks);
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
                                         Pair<T> topk[],  // T** kth,
                                         int* beam, int* k, const int tid,
                                         const int warp) {
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
    // NOTE(zcd): temporary solution
    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, true);

    if (maxid[0] / 32 == warp) {
      if (CudaShuffleSync(mask, *beam, (maxid[0]) % 32, 32) == MaxLength) break;
    }
  }
  return ret;
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

void k_select(const float* input, int count, void* encode, int* scan, int k,
              cudaStream_t stream) {
  const float sr = static_cast<float>(k) / count;
  if (sr > 0.1) {
    return;
  }
  float* value = reinterpret_cast<float*>(scan) + MAX_BLOCKS * MAX_THREADS;
  const int max_length = 5;
  int sampk = min(32, k / 2);
  const float sample_prop = static_cast<float>(k) / sampk;
  int sampdim = count / sample_prop;

  const int BlockSize = 1024;
  int threads = BlockSize;
  int blocks = min(16, static_cast<int>(sample_prop / 2));
  int lds = count / blocks;

  KeGetSampleTopK<float, max_length, BlockSize><<<blocks, threads, 0, stream>>>(
      value, input, lds, sampdim, sampk);
  KeGetTotalTopk<float, BlockSize><<<blocks, threads, 0, stream>>>(value,
                                                                   blocks);

  getNumBlocksAndThreads(count, MAX_BLOCKS, MAX_THREADS, blocks, threads);

  int smemSize = sizeof(float) * threads;
  int p_threads = min(blocks, threads);
  int p_blocks = iDivUp(blocks, p_threads);

  int* part = scan + threads * blocks + 1;
  float* threshold = reinterpret_cast<float*>(value);
  KeGetThreadCountByThreshold<float><<<blocks, threads, smemSize, 0>>>(
      input, scan, count, threshold);
  KePrefixSum<<<blocks, threads, smemSize, 0>>>(scan, 32, part);
  KePrefixSum<<<p_blocks, p_threads, smemSize, 0>>>(part, 32);
  KeGlobalPrefixSum<<<blocks - 1, threads>>>(scan + threads, part, count, k);
  int* index = reinterpret_cast<int*>(encode);
  value = reinterpret_cast<float*>(encode) + k;
  KeEncode<float><<<blocks, threads, 0, stream>>>(input, count, scan, value,
                                                  index, threshold, k);
}
