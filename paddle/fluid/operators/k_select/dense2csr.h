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

//#define FABS(a) ((a!=-INFINITY) ? fabs(a) : a)
#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_BLOCK 32
#define NUM_WARP (BLOCK_SIZE / WARP_SIZE)  // number of wraps
#define COLUMNS 65536                      // one wrap process one line
#define BLOCK_SCAN_COUNT (BLOCK_SIZE * 4)

__device__ inline int warp_scan(int value, int lane_id) {
#pragma unroll
  for (int i = 1; i <= WARP_SIZE; i *= 2) {
    unsigned int mask = 0xffffffff;
    int n = CudaShuffleUpSync(mask, value, i, WARP_SIZE);
    if (lane_id >= i) value += n;
  }
  return value;
}

__device__ inline int block_warp_scan(int value, int lane_id) {
#pragma unroll
  for (int i = 1; i <= NUM_WARP; i *= 2) {
    unsigned int mask = (1 << NUM_WARP) - 1;
    int n = CudaShuffleUpSync(mask, value, i, NUM_WARP);
    if (lane_id >= i) value += n;
  }
  return value;
}

__device__ inline int4 scan4(int4 idata4, volatile int* sdata, int lane_id,
                             int warp_id, int pre_sum = 0) {
  // step 1, get thread inclusive scan value
  idata4.y += idata4.x;
  idata4.z += idata4.y;
  idata4.w += idata4.z;

  // step 2, get warp inclusive scan value
  int oval = warp_scan(idata4.w, lane_id);

  // save warp scan into share memory
  if (threadIdx.x % WARP_SIZE == WARP_SIZE - 1) sdata[warp_id] = oval;
  __syncthreads();

  // step 3, get block inclusive scan value
  if (warp_id == 0 && lane_id < NUM_WARP) {
    sdata[lane_id] = block_warp_scan(sdata[lane_id], lane_id);
  }
  __syncthreads();

  // exclusive scan value of current warp
  int block_sum = 0;
  if (warp_id > 0) block_sum = sdata[warp_id - 1];

  // exclusive scan value of current thread
  oval = oval - idata4.w + block_sum + pre_sum;

  idata4.x += oval;
  idata4.y += oval;
  idata4.z += oval;
  idata4.w += oval;
  return idata4;
}

__device__ inline void scan_arbitrary(int4* output4, int4* input4, int* sdata,
                                      int n, int pre_sum = 0) {
  int tid = threadIdx.x;
  int lane_id = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  int4 idata4;
  if (tid < n / 4) {
    idata4 = input4[tid];
  } else if (tid == n / 4) {
    int t[4] = {0, 0, 0, 0};
    for (int i = 0; i < n % 4; ++i) {
      t[i] = *(reinterpret_cast<int*>(input4 + tid) + i);
    }
    idata4.x = t[0];
    idata4.y = t[1];
    idata4.z = t[2];
    idata4.w = t[3];
  } else {
    idata4 = (int4){0, 0, 0, 0};
  }

  idata4 = scan4(idata4, sdata, lane_id, warp_id, pre_sum);

  if (tid < n / 4) {
    output4[tid] = idata4;
  } else if (tid == n / 4) {
    int t[4] = {idata4.x, idata4.y, idata4.z, idata4.w};
    for (int i = 0; i < n % 4; ++i) {
      *(reinterpret_cast<int*>(output4 + tid) + i) = t[i];
    }
  }
}

__launch_bounds__(BLOCK_SIZE, 1) __global__
    void scan_all(int* output, int* input, int n) {
  __shared__ int sdata[NUM_WARP];
  __shared__ volatile int pre_sum;

  int tid = threadIdx.x;
  int lane_id = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  if (threadIdx.x == BLOCK_SIZE - 1) pre_sum = 0;

  int4* input4 = reinterpret_cast<int4*>(input);
  int4* output4 = reinterpret_cast<int4*>(output);
  int4* input4_end = input4 + (n / BLOCK_SCAN_COUNT) * BLOCK_SIZE;

  // full loop
  while (input4 < input4_end) {
    __syncthreads();
    int4 idata4 = input4[tid];
    idata4 = scan4(idata4, sdata, lane_id, warp_id, pre_sum);

    if (threadIdx.x == BLOCK_SIZE - 1) pre_sum = idata4.w;
    output4[tid] = idata4;

    input4 += BLOCK_SIZE;
    output4 += BLOCK_SIZE;
  }
  // remain loop
  n %= BLOCK_SCAN_COUNT;
  if (n == 0) return;
  __syncthreads();
  scan_arbitrary(output4, input4, sdata, n, pre_sum);
}

template <typename T>
__global__ void KeGetCountByThreshold(int* rowptr, const T* dense,
                                      const T* threshold, int* odata,
                                      int count) {
  int rows = DIVUP(count, COLUMNS);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int wid = tid / WARP_SIZE;
  int t = tid % WARP_SIZE;  // Thread (inside the warp)

  const T* dense_end = dense + count;
  // one block process NUM_WARP*COLUMNS dense data
  const int inc = MAX_BLOCK * NUM_WARP * COLUMNS;
  const int offset = wid * COLUMNS + t;
  dense += offset;
  T kth = *threshold;

  if (tid == 0) rowptr[0] = 0;

  // full loop
  while (wid < rows - 1) {
    int cnt = 0;
    for (int u = 0; u < COLUMNS / WARP_SIZE; ++u) {
      if (FABS(dense[u * WARP_SIZE]) >= kth) ++cnt;
    }
    // get warp inclusive scan
    int sum = warp_scan(cnt, t);
    odata[tid] = sum - cnt;
    if (t == WARP_SIZE - 1) rowptr[wid + 1] = sum;
    dense += inc;
    wid += MAX_BLOCK * NUM_WARP;
    tid += BLOCK_SIZE * MAX_BLOCK;
  }

  // remain loop
  if (wid == rows - 1) {
    int cnt = 0;
    int remain = count % COLUMNS;
    if (remain == 0) {
      for (int u = 0; u < COLUMNS / WARP_SIZE; ++u) {
        if (FABS(dense[u * WARP_SIZE]) >= kth) ++cnt;
      }
    } else {
      for (int u = 0; dense + u * WARP_SIZE < dense_end; ++u) {
        if (FABS(dense[u * WARP_SIZE]) >= kth) ++cnt;
      }
    }
    int sum = warp_scan(cnt, t);
    odata[tid] = sum - cnt;
    if (t == WARP_SIZE - 1) rowptr[wid + 1] = sum;
  }
}

template <typename T>
__global__ void KeEncodeCSR(int* index, T* value, const int* rowptr,
                            const T* dense, T* threshold, int* buff, int count,
                            int k) {
  int rows = DIVUP(count, COLUMNS);

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int wid = tid / WARP_SIZE;
  int t = tid % WARP_SIZE;  // Thread (inside the warp)

  const T* dense_end = dense + count;
  // one block process NUM = NUM_WARP * COLUMNS dense data
  const int inc = MAX_BLOCK * NUM_WARP * COLUMNS;
  const int offset = wid * COLUMNS + t;
  dense += offset;
  T kth = *threshold;

  // full loop
  while (wid < rows - 1) {
    int offset = rowptr[wid] + buff[tid];
    for (int u = 0; u < COLUMNS / WARP_SIZE; ++u) {
      T val = dense[u * WARP_SIZE];
      if (offset < k && FABS(val) >= kth) {
        index[offset] = u * WARP_SIZE + t;
        value[offset] = val;
        offset++;
      }
    }
    dense += inc;
    wid += MAX_BLOCK * NUM_WARP;
    tid += BLOCK_SIZE * MAX_BLOCK;
  }

  // remain loop
  if (wid == rows - 1) {
    int offset = rowptr[wid] + buff[tid];
    int remain = count % COLUMNS;
    if (remain == 0) {
      for (int u = 0; u < COLUMNS / WARP_SIZE; ++u) {
        T val = dense[u * WARP_SIZE];
        if (offset < k && FABS(val) >= kth) {
          index[offset] = u * WARP_SIZE + t;
          value[offset] = val;
          offset++;
        }
      }
    } else {
      for (int u = 0; dense + u * WARP_SIZE < dense_end; ++u) {
        T val = dense[u * WARP_SIZE];
        if (offset < k && FABS(val) >= kth) {
          index[offset] = u * WARP_SIZE + t;
          value[offset] = val;
          offset++;
        }
      }
    }  // end else
  }    // end if
}

template <typename T>
void dense2csr(void* encode, T* dense, T* threshold, int* buff, int count,
               int k, cudaStream_t stream) {
  int rows = DIVUP(count, COLUMNS);
  int blocks = DIVUP(rows, NUM_WARP);

  int* rowptr = static_cast<int*>(encode);
  KeGetCountByThreshold<T><<<min(MAX_BLOCK, blocks), BLOCK_SIZE, 0, stream>>>(
      rowptr, dense, threshold, buff, count);

  scan_all<<<1, BLOCK_SIZE, 0, stream>>>(rowptr, rowptr, rows + 1);

  int* index = rowptr + rows + 1;
  T* value = reinterpret_cast<T*>(index + k);
  KeEncodeCSR<T><<<min(MAX_BLOCK, blocks), BLOCK_SIZE, 0, stream>>>(
      index, value, rowptr, dense, threshold, buff, count, k);
}
