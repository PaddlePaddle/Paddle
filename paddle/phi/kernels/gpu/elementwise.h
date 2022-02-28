/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/gpu/reduce.h"

#ifdef __HIPCC__
constexpr int ELEMWISE_MAX_BLOCK_DIM = 256;
#else
constexpr int ELEMWISE_MAX_BLOCK_DIM = 1024;
#endif
#define BLOCK_X 32
#define BLOCK_Y 32

#define GetDivMod(dividend, divisor, div, mod) \
  do {                                         \
    const auto dividend_copy = dividend;       \
    *div = dividend_copy / divisor;            \
    *mod = dividend_copy % divisor;            \
  } while (0)

namespace phi {

// General binary elementwise comutaion with the support of broadcast.
template <typename Functor, typename T, typename OutType = T>
void ElementwiseCompute(const GPUContext &dev_ctx,
                        const DenseTensor &x,
                        const DenseTensor &y,
                        int axis,
                        Functor func,
                        DenseTensor *z) {
  std::vector<const DenseTensor *> ins = {&x, &y};
  std::vector<DenseTensor *> outs = {z};
  z->mutable_data<OutType>(dev_ctx.GetPlace());
  phi::funcs::BroadcastKernel<ElementwiseType::kBinary, T, OutType>(
      dev_ctx, ins, &outs, axis, func);
}

// Suppose only has contiguous dims
static inline bool CheckContiguousDims(const std::vector<int> &broadcast_pos) {
  for (int i = 1; i < broadcast_pos.size(); ++i) {
    if (broadcast_pos[i] != broadcast_pos[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

inline void ComputeBroadcastTranspositionArray(const int *x_one_indexs,
                                               int *x_trans_indexs,
                                               const int max_dim,
                                               const int x_one_size) {
  int diff = max_dim - x_one_size;
  std::copy_n(x_one_indexs, x_one_size, x_trans_indexs + diff);
  int p = 0;
  int q = diff;
  for (int i = 0; i < max_dim; ++i) {
    if (q < max_dim && i == x_trans_indexs[q]) {
      ++q;
    } else {
      x_trans_indexs[p++] = i;
    }
  }
}

// Check input can be split into 2 parts
static inline bool SplitDims(const std::vector<int> &y_broadcast_pos,
                             int max_dim) {
  bool can_split_dim2 = true;
  // must at start or end.
  if (y_broadcast_pos[0] != 0 &&
      y_broadcast_pos[y_broadcast_pos.size() - 1] != max_dim - 1) {
    can_split_dim2 = false;
  } else {
    for (int i = 1; i < y_broadcast_pos.size(); ++i) {
      // dim must be continue
      if (y_broadcast_pos[i] != y_broadcast_pos[i - 1] + 1) {
        can_split_dim2 = false;
        break;
      }
    }
  }
  return can_split_dim2;
}

inline void ComputeBroadcastKernelSize(int *x_dims_array,
                                       int *out_dims_array,
                                       int *x_blocks,
                                       int *x_threads,
                                       int max_dim) {
  *x_blocks = 1;
  *x_threads = 1;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] == out_dims_array[i]) {
      *x_blocks *= x_dims_array[i];
    } else {
      *x_threads *= out_dims_array[i];
    }
  }
}

#ifndef __xpu__
template <typename T, typename OP, typename Tout = T>
static __global__ void FastCommonGradBroadcastOneCUDAKernel(const T *x,
                                                            const T *y,
                                                            const Tout *out,
                                                            const Tout *dout,
                                                            int pre,
                                                            int n,
                                                            int post,
                                                            int y_pre,
                                                            int y_n,
                                                            int y_post,
                                                            bool is_xsize,
                                                            OP op,
                                                            T *dd) {
  int tid = THREAD_ID_X;
  int bid = BLOCK_ID_X;

  T val(0);
  if (is_xsize) {
    // do reduce for x
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int x_offset = b_i * n * post + b_j;
      int out_offset = b_i * n * post + i * post + b_j;

      // Get y pre rows id with x post and y_pre.
      int b_yi = bid / (post * y_pre);
      int b_yj = bid % y_post;
      int y_offset = b_yi * y_n + i * y_post + b_yj;

      if (dd) {
        val += op(x[x_offset], y[y_offset], out[out_offset], dout[out_offset]);
      }
    }
    if (dd) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dd[bid] = val;
      }
    }
  } else {
    // do reduce for y
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int y_offset = b_i * n * post + b_j;
      int out_offset = b_i * n * post + i * post + b_j;

      int b_yi = bid / (post * y_pre);
      int b_yj = bid % y_post;
      int x_offset = b_yi * y_n + i * y_post + b_yj;

      if (dd) {
        val += op(x[x_offset], y[y_offset], out[out_offset], dout[out_offset]);
      }
    }
    if (dd) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dd[bid] = val;
      }
    }
  }
}

template <typename T, typename DY_OP, typename DX_OP, typename Tout = T>
static __global__ void FastCommonGradBroadcastAllCUDAKernel(
    const T *x,
    const T *y,
    const Tout *out,
    const Tout *dout,
    int pre,
    int n,
    int post,
    bool is_xsize_larger,
    DX_OP dx_op,
    DY_OP dy_op,
    T *dx,
    T *dy) {
  int tid = THREAD_ID_X;
  int bid = BLOCK_ID_X;

  T val(0);
  if (is_xsize_larger) {
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int x_offset = b_i * n * post + i * post + b_j;
      int y_offset = b_i * post + b_j;
      if (dx) {
        dx[x_offset] =
            dx_op(x[x_offset], y[y_offset], out[x_offset], dout[x_offset]);
      }
      if (dy) {
        val += dy_op(x[x_offset], y[y_offset], out[x_offset], dout[x_offset]);
      }
    }
    if (dy) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dy[bid] = val;
      }
    }
  } else {
    for (int i = tid; i < n; i += ELEMWISE_MAX_BLOCK_DIM) {
      int b_i = bid / post;
      int b_j = bid % post;
      int y_offset = b_i * n * post + i * post + b_j;
      int x_offset = b_i * post + b_j;
      if (dy) {
        dy[y_offset] =
            dy_op(x[x_offset], y[y_offset], out[y_offset], dout[y_offset]);
      }
      if (dx) {
        val += dx_op(x[x_offset], y[y_offset], out[y_offset], dout[y_offset]);
      }
    }
    if (dx) {
      int h = n > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : n;
      val = paddle::platform::reduceSum(val, tid, h);
      if (tid == 0) {
        dx[bid] = val;
      }
    }
  }
}

template <typename T, typename DY_OP, typename Tout = T>
static __global__ void FastCommonGradBroadcastCUDAKernelHeight(const T *x,
                                                               const T *y,
                                                               const Tout *out,
                                                               const Tout *dout,
                                                               int h,
                                                               int w,
                                                               DY_OP dy_op,
                                                               T *dy,
                                                               int x_h,
                                                               int x_w,
                                                               bool is_y) {
  __shared__ T sdata[BLOCK_Y][BLOCK_X + 1];

  T val(0);
  size_t width_stride = GRID_NUM_X * BLOCK_NUM_X;
  size_t idx = THREAD_ID_X + BLOCK_NUM_X * BLOCK_ID_X;
  size_t full_width =
      (w & (~((uint64_t)(BLOCK_X - 1)))) + ((w & (BLOCK_X - 1)) ? BLOCK_X : 0);
  size_t full_height =
      (h & (~((uint64_t)(BLOCK_Y - 1)))) + ((h & (BLOCK_Y - 1)) ? BLOCK_Y : 0);
  if (is_y) {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[THREAD_ID_Y][THREAD_ID_X] = 0;
      for (int n = THREAD_ID_Y; n < full_height; n += BLOCK_Y) {
        int out_offset = n * w + m;
        int x_offset = (n % x_h) * x_w + m % x_w;
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[x_offset], y[m], out[out_offset], dout[out_offset]);
            sdata[THREAD_ID_Y][THREAD_ID_X] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[THREAD_ID_X][THREAD_ID_Y];
        for (int i = warpSize >> 1; i > 0; i >>= 1) {
          my_val += paddle::platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        }
        __syncthreads();
        if ((THREAD_ID_X == 0)) {
          sdata[0][THREAD_ID_Y] = my_val;
        }
        __syncthreads();
        if (THREAD_ID_Y == 0 && m < w) {
          dy[m] = sdata[0][THREAD_ID_X];
        }
      }
    }
  } else {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[THREAD_ID_Y][THREAD_ID_X] = 0;
      for (int n = THREAD_ID_Y; n < full_height; n += BLOCK_Y) {
        int out_offset = n * w + m;
        int y_offset = (n % x_h) * x_w + m % x_w;
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[m], y[y_offset], out[out_offset], dout[out_offset]);
            sdata[THREAD_ID_Y][THREAD_ID_X] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[THREAD_ID_X][THREAD_ID_Y];
        for (int i = warpSize >> 1; i > 0; i >>= 1) {
          my_val += paddle::platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        }
        __syncthreads();
        if ((THREAD_ID_X == 0)) {
          sdata[0][THREAD_ID_Y] = my_val;
        }
        __syncthreads();
        if (THREAD_ID_Y == 0 && m < w) {
          dy[m] = sdata[0][THREAD_ID_X];
        }
      }
    }
  }
}

template <typename T, typename DY_OP, typename Tout = T>
static __global__ void CommonGradBroadcast1CUDAKernelHeight(const T *x,
                                                            const T *y,
                                                            const Tout *out,
                                                            const Tout *dout,
                                                            int h,
                                                            int w,
                                                            DY_OP dy_op,
                                                            T *dy,
                                                            int x_h,
                                                            int x_w,
                                                            bool is_y) {
  int j = BLOCK_ID_X;
  int i = THREAD_ID_X;
  int tid = THREAD_ID_X;
  T val(0);

  if (is_y) {
    do {
      int out_offset = i * w + j;
      int x_offset = (i % x_h) * x_w + j % x_w;
      if (dy) {
        val += dy_op(x[x_offset], y[j], out[out_offset], dout[out_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dy[j] = val;
      }
    }
  } else {
    do {
      int out_offset = i * w + j;
      int y_offset = (i % x_h) * x_w + j % x_w;
      if (dy) {
        val += dy_op(x[j], y[y_offset], out[out_offset], dout[out_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dy[j] = val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static __global__ void ElemwiseGradBroadcast1CUDAKernel(const T *x,
                                                        const T *y,
                                                        const Tout *out,
                                                        const Tout *dout,
                                                        int h,
                                                        int w,
                                                        bool is_xsize_larger,
                                                        DX_OP dx_op,
                                                        DY_OP dy_op,
                                                        T *dx,
                                                        T *dy) {
  int j = BLOCK_ID_X;
  int i = THREAD_ID_X;
  int tid = THREAD_ID_X;
  T val(0);
  if (is_xsize_larger) {
    do {
      int x_offset = i * w + j;
      if (dx) {
        dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }
      if (dy) {
        val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dy) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dy[j] = val;
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    do {
      int y_offset = i * w + j;
      if (dy) {
        dy[y_offset] = dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }
      if (dx) {
        val += dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }
      i += ELEMWISE_MAX_BLOCK_DIM;
    } while (i < h);

    if (dx) {
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dx[j] = val;
      }
    }
  }
}

// suppose use 2D block is fast because more parallel
// and memory coalesced
template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static __global__ void FastElemwiseGradBroadcast1CUDAKernel(
    const T *x,
    const T *y,
    const Tout *out,
    const Tout *dout,
    int h,
    int w,
    bool is_xsize_larger,
    DX_OP dx_op,
    DY_OP dy_op,
    T *dx,
    T *dy) {
  __shared__ T sdata[BLOCK_Y][BLOCK_X + 1];

  T val(0);
  size_t width_stride = GRID_NUM_X * BLOCK_NUM_X;
  size_t idx = THREAD_ID_X + BLOCK_NUM_X * BLOCK_ID_X;
  size_t full_width =
      (w & (~((uint64_t)(BLOCK_X - 1)))) + ((w & (BLOCK_X - 1)) ? BLOCK_X : 0);
  size_t full_height =
      (h & (~((uint64_t)(BLOCK_Y - 1)))) + ((h & (BLOCK_Y - 1)) ? BLOCK_Y : 0);
  if (is_xsize_larger) {
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[THREAD_ID_Y][THREAD_ID_X] = 0;
      for (int n = THREAD_ID_Y; n < full_height; n += BLOCK_Y) {
        int x_offset = n * w + m;
        if (dx && m < w && n < h) {
          dx[x_offset] =
              dx_op(x[x_offset], y[m], out[x_offset], dout[x_offset]);
        }
        if (dy) {
          if (m < w && n < h) {
            T val = dy_op(x[x_offset], y[m], out[x_offset], dout[x_offset]);
            sdata[THREAD_ID_Y][THREAD_ID_X] += val;
          }
          __syncthreads();
        }
      }
      if (dy) {
        T my_val = sdata[THREAD_ID_X][THREAD_ID_Y];
        for (int i = warpSize >> 1; i > 0; i >>= 1)
          my_val += paddle::platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        __syncthreads();
        if ((THREAD_ID_X == 0)) {
          sdata[0][THREAD_ID_Y] = my_val;
        }
        __syncthreads();
        if (THREAD_ID_Y == 0 && m < w) {
          dy[m] = sdata[0][THREAD_ID_X];
        }
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    for (int m = idx; m < full_width; m += width_stride) {
      sdata[THREAD_ID_Y][THREAD_ID_X] = 0;
      for (int n = THREAD_ID_Y; n < full_height; n += BLOCK_Y) {
        int y_offset = n * w + m;
        if (dy && m < w && n < h) {
          dy[y_offset] =
              dy_op(x[m], y[y_offset], out[y_offset], dout[y_offset]);
        }
        if (dx) {
          if (m < w && n < h) {
            T val = dx_op(x[m], y[y_offset], out[y_offset], dout[y_offset]);
            sdata[THREAD_ID_Y][THREAD_ID_X] += val;
          }
          __syncthreads();
        }
      }
      if (dx) {
        T my_val = sdata[THREAD_ID_X][THREAD_ID_Y];
        for (int i = warpSize >> 1; i > 0; i >>= 1)
          my_val += paddle::platform::CudaShuffleXorSync(0xFFFFFFFF, my_val, i);
        __syncthreads();
        if ((THREAD_ID_X == 0)) {
          sdata[0][THREAD_ID_Y] = my_val;
        }
        __syncthreads();
        if (THREAD_ID_Y == 0 && m < w) {
          dx[m] = sdata[0][THREAD_ID_X];
        }
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static __global__ void ElemwiseGradBroadcast2CUDAKernel(const T *x,
                                                        const T *y,
                                                        const Tout *out,
                                                        const Tout *dout,
                                                        int pre,
                                                        int n,
                                                        int post,
                                                        bool is_xsize_larger,
                                                        DX_OP dx_op,
                                                        DY_OP dy_op,
                                                        T *dx,
                                                        T *dy) {
  int tid = THREAD_ID_X;
  int j = BLOCK_ID_X;

  T val(0);
  int ttid = tid;

  if (is_xsize_larger) {
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int x_offset = i * n * post + j * post + k;

      if (dx != nullptr) {
        dx[x_offset] = dx_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }

      if (dy != nullptr) {
        val += dy_op(x[x_offset], y[j], out[x_offset], dout[x_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dy) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dy[j] = val;
      }
    }
  } else {  // x.dims < y.dims, broadcast for x.
    while (true) {
      int i = ttid / post;
      int k = ttid % post;
      if (i >= pre) break;

      int y_offset = i * n * post + j * post + k;

      if (dy != nullptr) {
        dy[y_offset] = dy_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }

      if (dx != nullptr) {
        val += dx_op(x[j], y[y_offset], out[y_offset], dout[y_offset]);
      }

      ttid += ELEMWISE_MAX_BLOCK_DIM;
    }

    if (dx) {
      int h = pre * post;
      h = h > ELEMWISE_MAX_BLOCK_DIM ? ELEMWISE_MAX_BLOCK_DIM : h;
      val = paddle::platform::reduceSum(val, tid, h);
      if (THREAD_ID_X == 0) {
        dx[j] = val;
      }
    }
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static void ElemwiseGradBroadcast1CUDA(gpuStream_t stream,
                                       const T *x,
                                       const T *y,
                                       const Tout *out,
                                       const Tout *dout,
                                       int h,
                                       int w,
                                       bool is_xsize_larger,
                                       DX_OP dx_op,
                                       DY_OP dy_op,
                                       T *dx,
                                       T *dy) {
  // For small case use 1D block
  constexpr int half_walf = 16;
  if (w < half_walf || h < half_walf) {
    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
    int gird_size = w;
    ElemwiseGradBroadcast1CUDAKernel<<<gird_size, block_size, 0, stream>>>(
        x, y, out, dout, h, w, is_xsize_larger, dx_op, dy_op, dx, dy);
  } else {
    // suppose perfoemance improves with h increased.
    dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
    int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
    FastElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0, stream>>>(
        x, y, out, dout, h, w, is_xsize_larger, dx_op, dy_op, dx, dy);
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
static void ElemwiseGradBroadcast2CUDA(gpuStream_t stream,
                                       const T *x,
                                       const T *y,
                                       const Tout *out,
                                       const Tout *dout,
                                       int pre,
                                       int n,
                                       int post,
                                       bool is_xsize_larger,
                                       DX_OP dx_op,
                                       DY_OP dy_op,
                                       T *dx,
                                       T *dy) {
  int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, pre * post);
  int gird_size = n;
  ElemwiseGradBroadcast2CUDAKernel<<<gird_size, block_size, 0, stream>>>(
      x, y, out, dout, pre, n, post, is_xsize_larger, dx_op, dy_op, dx, dy);
}

template <typename T, typename DX_OP, typename Tout = T>
__global__ void CommonGradBroadcastCUDAKernel(const int *x_strides_array,
                                              const int *y_strides_array,
                                              const int *out_dims_array,
                                              const int *y_strides_order,
                                              const int *y_dims_order,
                                              const T *x,
                                              const T *y,
                                              const Tout *out,
                                              const Tout *dout,
                                              T *dx,
                                              int out_size,
                                              int max_dim,
                                              int thread_num,
                                              DX_OP dx_op) {
  T val(0);
  int i = BLOCK_ID_X;
  int tid = THREAD_ID_X;
  for (int j = tid; j < thread_num; j += BLOCK_NUM_X) {
    const int X_index = i * thread_num + j;
    int out_index = X_index;
    int C_index = 0;
    int B_index = i * thread_num + j;
    int remainder = 0;
#pragma unroll
    for (int d = max_dim - 1; d >= 0; --d) {
      GetDivMod(B_index, y_dims_order[d], &B_index, &remainder);
      C_index += remainder * y_strides_order[d];
    }
    int x_index = 0;
    int y_index = 0;
    int C_index_val = C_index;
#pragma unroll
    for (int d = max_dim - 1; d >= 0; --d) {
      GetDivMod(C_index_val, out_dims_array[d], &C_index_val, &remainder);
      x_index += remainder * x_strides_array[d];
      y_index += remainder * y_strides_array[d];
    }
    out_index = C_index;
    val += dx_op(x[x_index], y[y_index], out[out_index], dout[out_index]);
  }
  val = paddle::platform::reduceSum(val, tid, thread_num);
  if (THREAD_ID_X == 0) {
    dx[i] = val;
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonGradBroadcastCUDA(const DenseTensor &x,
                             const DenseTensor &y,
                             const DenseTensor &out,
                             const DenseTensor &dout,
                             DenseTensor *dx,
                             DenseTensor *dy,
                             int *x_dims_array,
                             int *y_dims_array,
                             int *out_dims_array,
                             int max_dim,
                             const GPUContext &ctx,
                             DX_OP dx_op,
                             DY_OP dy_op) {
  const auto gplace = ctx.GetPlace();
  auto cplace = phi::CPUPlace();
  const T *x_data = x.data<T>();
  const T *y_data = y.data<T>();
  const Tout *out_data = out.data<Tout>();
  const Tout *dout_data = dout.data<Tout>();
  T *dx_data = dx == nullptr ? nullptr : ctx.Alloc<T>(dx);
  T *dy_data = dy == nullptr ? nullptr : ctx.Alloc<T>(dy);

  std::vector<int> x_one_indexs;
  std::vector<int> y_one_indexs;
  for (int i = 0; i < max_dim; i++) {
    if (x_dims_array[i] != y_dims_array[i]) {
      if (x_dims_array[i] == 1) {
        x_one_indexs.push_back(i);
      }
      if (y_dims_array[i] == 1) {
        y_one_indexs.push_back(i);
      }
    }
  }

  std::vector<int> x_trans_indexs(max_dim);
  std::vector<int> y_trans_indexs(max_dim);
  ComputeBroadcastTranspositionArray(
      x_one_indexs.data(), x_trans_indexs.data(), max_dim, x_one_indexs.size());
  ComputeBroadcastTranspositionArray(
      y_one_indexs.data(), y_trans_indexs.data(), max_dim, y_one_indexs.size());

  // compute array stride for cuda kernel;
  // e.g. x.dims=[2,3,4], x_stride=[12,4,1]
  std::vector<int> x_strides_array(max_dim);
  std::vector<int> y_strides_array(max_dim);
  std::vector<int> out_strides_array(max_dim);
  int x_stride = 1;
  int y_stride = 1;
  int z_stride = 1;
  for (int i = max_dim - 1; i >= 0; i--) {
    x_strides_array[i] = x_dims_array[i] == 1 ? 0 : x_stride;
    y_strides_array[i] = y_dims_array[i] == 1 ? 0 : y_stride;
    out_strides_array[i] = z_stride;
    x_stride *= x_dims_array[i];
    y_stride *= y_dims_array[i];
    z_stride *= out_dims_array[i];
  }

  std::vector<int> x_strides_order(max_dim);
  std::vector<int> y_strides_order(max_dim);
  std::vector<int> x_dims_order(max_dim);
  std::vector<int> y_dims_order(max_dim);
  for (int i = 0; i < max_dim; ++i) {
    x_strides_order[i] = out_strides_array[x_trans_indexs[i]];
    y_strides_order[i] = out_strides_array[y_trans_indexs[i]];
    x_dims_order[i] = out_dims_array[x_trans_indexs[i]];
    y_dims_order[i] = out_dims_array[y_trans_indexs[i]];
  }
  std::vector<int> x_broadcast_pos;
  std::vector<int> y_broadcast_pos;

  int bytes = max_dim * sizeof(int);

  for (int i = 0; i < max_dim; ++i) {
    if (x_dims_array[i] != out_dims_array[i] && x_dims_array[i] == 1) {
      x_broadcast_pos.emplace_back(i);
    }
    if (y_dims_array[i] != out_dims_array[i] && y_dims_array[i] == 1) {
      y_broadcast_pos.emplace_back(i);
    }
  }

  auto stream = ctx.stream();
  bool can_split_x = false;
  bool can_split_y = false;

  auto FastCommonCUDAF = [&](const std::vector<int> &broadcast_pos, bool is_y) {
    int h = std::accumulate(out_dims_array,
                            out_dims_array + broadcast_pos.size(),
                            1,
                            std::multiplies<int>());
    int w = std::accumulate(out_dims_array + broadcast_pos.size(),
                            out_dims_array + max_dim,
                            1,
                            std::multiplies<int>());

    VLOG(3) << "FastCommonCUDAF elementwise w:" << w << " h:" << h
            << " is_y:" << is_y;

    int split_h;
    int split_w;
    int kh = h;
    int kw = w;

    if (is_y) {
      split_h = std::accumulate(x_dims_array,
                                x_dims_array + broadcast_pos.size(),
                                1,
                                std::multiplies<int>());
      split_w = std::accumulate(x_dims_array + broadcast_pos.size(),
                                x_dims_array + max_dim,
                                1,
                                std::multiplies<int>());

    } else {
      split_h = std::accumulate(y_dims_array,
                                y_dims_array + broadcast_pos.size(),
                                1,
                                std::multiplies<int>());
      split_w = std::accumulate(y_dims_array + broadcast_pos.size(),
                                y_dims_array + max_dim,
                                1,
                                std::multiplies<int>());
    }

    if (h > split_h) kh = split_h;
    if (w > split_w) kw = split_w;

    if (is_y) {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        CommonGradBroadcast1CUDAKernelHeight<<<grid_size,
                                               block_size,
                                               0,
                                               stream>>>(x_data,
                                                         y_data,
                                                         out_data,
                                                         dout_data,
                                                         h,
                                                         w,
                                                         dy_op,
                                                         dy_data,
                                                         kh,
                                                         kw,
                                                         is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
        FastCommonGradBroadcastCUDAKernelHeight<<<grid_size,
                                                  block_size,
                                                  0,
                                                  stream>>>(x_data,
                                                            y_data,
                                                            out_data,
                                                            dout_data,
                                                            h,
                                                            w,
                                                            dy_op,
                                                            dy_data,
                                                            kh,
                                                            kw,
                                                            is_y);
      }
    } else {
      if (w < 16 || h < 16) {
        int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
        int grid_size = w;
        CommonGradBroadcast1CUDAKernelHeight<<<grid_size,
                                               block_size,
                                               0,
                                               stream>>>(x_data,
                                                         y_data,
                                                         out_data,
                                                         dout_data,
                                                         h,
                                                         w,
                                                         dx_op,
                                                         dx_data,
                                                         kh,
                                                         kw,
                                                         is_y);
      } else {
        dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
        int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
        FastCommonGradBroadcastCUDAKernelHeight<<<grid_size,
                                                  block_size,
                                                  0,
                                                  stream>>>(x_data,
                                                            y_data,
                                                            out_data,
                                                            dout_data,
                                                            h,
                                                            w,
                                                            dx_op,
                                                            dx_data,
                                                            kh,
                                                            kw,
                                                            is_y);
      }
    }
  };

  auto FastBroadCastHeightCUDAF = [&](const std::vector<int> &broadcast_pos,
                                      bool x_large) {
    int h = std::accumulate(out_dims_array,
                            out_dims_array + broadcast_pos.size(),
                            1,
                            std::multiplies<int>());
    int w = std::accumulate(out_dims_array + broadcast_pos.size(),
                            out_dims_array + max_dim,
                            1,
                            std::multiplies<int>());

    VLOG(3) << "FastBroadCastHeightCUDAF w:" << w << " h:" << h;

    if (w < 16 || h < 16) {
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, h);
      int grid_size = w;
      ElemwiseGradBroadcast1CUDAKernel<<<grid_size, block_size, 0, stream>>>(
          x_data,
          y_data,
          out_data,
          dout_data,
          h,
          w,
          x_large,
          dx_op,
          dy_op,
          dx_data,
          dy_data);
    } else {
      dim3 block_size = dim3(BLOCK_X, BLOCK_Y);
      int grid_size = (w + BLOCK_X - 1) / BLOCK_X;
      FastElemwiseGradBroadcast1CUDAKernel<<<grid_size,
                                             block_size,
                                             0,
                                             stream>>>(x_data,
                                                       y_data,
                                                       out_data,
                                                       dout_data,
                                                       h,
                                                       w,
                                                       x_large,
                                                       dx_op,
                                                       dy_op,
                                                       dx_data,
                                                       dy_data);
    }
  };

  auto FastBroadCastAllCUDAF = [&](
      const std::vector<int> &broadcast_pos, int max_dim, bool is_x_large) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(
        out_dims_array, out_dims_array + axis, 1, std::multiplies<int>());
    int mid = 1;
    int post = 1;

    if (broadcast_pos.size() == 1) {
      mid = out_dims_array[axis];
      post = std::accumulate(out_dims_array + axis + 1,
                             out_dims_array + max_dim,
                             1,
                             std::multiplies<int>());
    } else {
      mid = std::accumulate(out_dims_array + axis,
                            out_dims_array + broadcast_pos.back() + 1,
                            1,
                            std::multiplies<int>());
      post = std::accumulate(out_dims_array + broadcast_pos.back() + 1,
                             out_dims_array + max_dim,
                             1,
                             std::multiplies<int>());
    }

    VLOG(3) << "FastBroadCastAllCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;

    int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
    int grid_size = pre * post;

    FastCommonGradBroadcastAllCUDAKernel<<<grid_size, block_size, 0, stream>>>(
        x_data,
        y_data,
        out_data,
        dout_data,
        pre,
        mid,
        post,
        is_x_large,
        dx_op,
        dy_op,
        dx_data,
        dy_data);
  };

  auto FastBroadCastOneCUDAF = [&](
      const std::vector<int> &broadcast_pos, int max_dim, bool is_x) {
    int axis = broadcast_pos[0];
    int pre = std::accumulate(
        out_dims_array, out_dims_array + axis, 1, std::multiplies<int>());
    int mid = out_dims_array[axis];
    int post = std::accumulate(out_dims_array + axis + 1,
                               out_dims_array + max_dim,
                               1,
                               std::multiplies<int>());

    int k_pre;
    int k_mid;
    int k_post;

    if (is_x) {
      k_pre = std::accumulate(
          y_dims_array, y_dims_array + axis, 1, std::multiplies<int>());
      k_mid = y_dims_array[axis];
      k_post = std::accumulate(y_dims_array + axis + 1,
                               y_dims_array + max_dim,
                               1,
                               std::multiplies<int>());
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
      int grid_size = pre * post;
      // we need to calc y offset with blockid, so do x_pre/y_pre to get left
      // size.
      if (k_pre != pre) k_pre = pre / k_pre;

      FastCommonGradBroadcastOneCUDAKernel<<<grid_size,
                                             block_size,
                                             0,
                                             stream>>>(x_data,
                                                       y_data,
                                                       out_data,
                                                       dout_data,
                                                       pre,
                                                       mid,
                                                       post,
                                                       k_pre,
                                                       k_mid,
                                                       k_post,
                                                       true,
                                                       dx_op,
                                                       dx_data);
    } else {
      k_pre = std::accumulate(
          x_dims_array, x_dims_array + axis, 1, std::multiplies<int>());
      k_mid = x_dims_array[axis];
      k_post = std::accumulate(x_dims_array + axis + 1,
                               x_dims_array + max_dim,
                               1,
                               std::multiplies<int>());
      int block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, mid);
      int grid_size = pre * post;
      if (k_pre != pre) k_pre = pre / k_pre;

      FastCommonGradBroadcastOneCUDAKernel<<<grid_size,
                                             block_size,
                                             0,
                                             stream>>>(x_data,
                                                       y_data,
                                                       out_data,
                                                       dout_data,
                                                       pre,
                                                       mid,
                                                       post,
                                                       k_pre,
                                                       k_mid,
                                                       k_post,
                                                       false,
                                                       dy_op,
                                                       dy_data);
    }
    VLOG(3) << "FastBroadCastOneCUDAF pre:" << pre << " mid:" << mid
            << " post:" << post;
  };

  // do fast elementwise if: 1. only one input need to do broadcast, we can
  // fallback
  // to old fast path.
  // 2. if both x and y need broadcast, then do it one by one.
  bool fast_broadcast = false;
  if (x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    can_split_y = SplitDims(y_broadcast_pos, max_dim);
    if (can_split_y) {
      // only y need to do broadcast on h
      if (y_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(y_broadcast_pos, true);
        fast_broadcast = true;
      }
    } else if (y_broadcast_pos.size() == 1 ||
               CheckContiguousDims(y_broadcast_pos)) {  // for only one dim and
                                                        // contiguous broadcast.
      // If cannot split,  which means input has 3 parts
      FastBroadCastAllCUDAF(y_broadcast_pos, max_dim, true);
      fast_broadcast = true;
    }
  } else if (y_broadcast_pos.empty() && !x_broadcast_pos.empty()) {
    // only x need broadcast
    can_split_x = SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastBroadCastHeightCUDAF(x_broadcast_pos, false);
        fast_broadcast = true;
      }
    } else if (x_broadcast_pos.size() == 1 ||
               CheckContiguousDims(x_broadcast_pos)) {
      FastBroadCastAllCUDAF(x_broadcast_pos, max_dim, false);
      fast_broadcast = true;
    }
  } else if (!x_broadcast_pos.empty() && !y_broadcast_pos.empty()) {
    // do x and y broadcast each.
    can_split_y = SplitDims(y_broadcast_pos, max_dim);
    bool fast_broadcast_x = false;
    bool fast_broadcast_y = false;
    if (can_split_y) {
      // begin at start.
      if (y_broadcast_pos[0] == 0) {
        FastCommonCUDAF(y_broadcast_pos, true);
        fast_broadcast_y = true;
      }
    } else if (y_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(y_broadcast_pos, max_dim, false);
      can_split_y = true;
      fast_broadcast_y = true;
    }
    can_split_x = SplitDims(x_broadcast_pos, max_dim);
    if (can_split_x) {
      if (x_broadcast_pos[0] == 0) {
        FastCommonCUDAF(x_broadcast_pos, false);
        fast_broadcast_x = true;
      }
    } else if (x_broadcast_pos.size() == 1) {
      FastBroadCastOneCUDAF(x_broadcast_pos, max_dim, true);
      can_split_x = true;
      fast_broadcast_x = true;
    }
    VLOG(3) << "CommonBroadcast can_split_y:" << can_split_y
            << " can_split_x:" << can_split_x;
    // if both x and y into fast path then return
    if (fast_broadcast_x && fast_broadcast_y) {
      fast_broadcast = true;
    }
    if (can_split_y && can_split_x && fast_broadcast) return;
  }

  // Should remove memory copy, use reg instead.
  if (fast_broadcast) {
    return;
  }
  int x_blocks = 0;
  int x_threads = 0;
  ComputeBroadcastKernelSize(
      x_dims_array, out_dims_array, &x_blocks, &x_threads, max_dim);
  int y_blocks = 0;
  int y_threads = 0;
  ComputeBroadcastKernelSize(
      y_dims_array, out_dims_array, &y_blocks, &y_threads, max_dim);

  auto x_strides_array_tmp = paddle::memory::Alloc(ctx, bytes);
  int *x_strides_array_gpu =
      reinterpret_cast<int *>(x_strides_array_tmp->ptr());
  paddle::memory::Copy(gplace,
                       x_strides_array_gpu,
                       cplace,
                       x_strides_array.data(),
                       bytes,
                       ctx.stream());

  auto y_strides_array_tmp = paddle::memory::Alloc(ctx, bytes);
  int *y_strides_array_gpu =
      reinterpret_cast<int *>(y_strides_array_tmp->ptr());
  paddle::memory::Copy(gplace,
                       y_strides_array_gpu,
                       cplace,
                       y_strides_array.data(),
                       bytes,
                       ctx.stream());

  auto out_dims_array_tmp = paddle::memory::Alloc(ctx, bytes);
  int *out_dims_array_gpu = reinterpret_cast<int *>(out_dims_array_tmp->ptr());
  paddle::memory::Copy(
      gplace, out_dims_array_gpu, cplace, out_dims_array, bytes, ctx.stream());

  const int out_size = std::accumulate(
      out_dims_array, out_dims_array + max_dim, 1, std::multiplies<int>());
  int x_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, x_threads);
  int y_block_size = std::min(ELEMWISE_MAX_BLOCK_DIM, y_threads);
  if (dx) {
    auto x_strides_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int *x_strides_order_gpu =
        reinterpret_cast<int *>(x_strides_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         x_strides_order_gpu,
                         cplace,
                         x_strides_order.data(),
                         bytes,
                         ctx.stream());

    auto x_dims_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int *x_dims_order_gpu = reinterpret_cast<int *>(x_dims_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         x_dims_order_gpu,
                         cplace,
                         x_dims_order.data(),
                         bytes,
                         ctx.stream());
    CommonGradBroadcastCUDAKernel<
        T,
        DX_OP,
        Tout><<<x_blocks, x_block_size, 0, ctx.stream()>>>(x_strides_array_gpu,
                                                           y_strides_array_gpu,
                                                           out_dims_array_gpu,
                                                           x_strides_order_gpu,
                                                           x_dims_order_gpu,
                                                           x_data,
                                                           y_data,
                                                           out_data,
                                                           dout_data,
                                                           dx_data,
                                                           out_size,
                                                           max_dim,
                                                           x_threads,
                                                           dx_op);
  }
  if (dy) {
    auto y_strides_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int *y_strides_order_gpu =
        reinterpret_cast<int *>(y_strides_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         y_strides_order_gpu,
                         cplace,
                         y_strides_order.data(),
                         bytes,
                         ctx.stream());

    auto y_dims_order_tmp = paddle::memory::Alloc(ctx, bytes);
    int *y_dims_order_gpu = reinterpret_cast<int *>(y_dims_order_tmp->ptr());
    paddle::memory::Copy(gplace,
                         y_dims_order_gpu,
                         cplace,
                         y_dims_order.data(),
                         bytes,
                         ctx.stream());
    CommonGradBroadcastCUDAKernel<
        T,
        DY_OP,
        Tout><<<y_blocks, y_block_size, 0, ctx.stream()>>>(x_strides_array_gpu,
                                                           y_strides_array_gpu,
                                                           out_dims_array_gpu,
                                                           y_strides_order_gpu,
                                                           y_dims_order_gpu,
                                                           x_data,
                                                           y_data,
                                                           out_data,
                                                           dout_data,
                                                           dy_data,
                                                           out_size,
                                                           max_dim,
                                                           y_threads,
                                                           dy_op);
  }
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void CommonElementwiseBroadcastBackward(const GPUContext &ctx,
                                        const DDim &x_dims,
                                        const DDim &y_dims,
                                        const DenseTensor &x,
                                        const DenseTensor &y,
                                        const DenseTensor &out,
                                        const DenseTensor &dout,
                                        int axis,
                                        DenseTensor *dx,
                                        DenseTensor *dy,
                                        DX_OP dx_op,
                                        DY_OP dy_op) {
  int max_dim = std::max(x_dims.size(), y_dims.size());
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  funcs::GetBroadcastDimsArrays(x_dims,
                                y_dims,
                                x_dims_array.data(),
                                y_dims_array.data(),
                                out_dims_array.data(),
                                max_dim,
                                axis);
  // for inplace strategy. memset will make dx and dout clear and get wrong
  // result.
  if (dx && dx->IsSharedBufferWith(dout)) {
    dx->clear();
    dx->mutable_data<T>(x_dims, ctx.GetPlace());
  }

  VLOG(3) << "CommonElementwiseBroadcastBackward xdims:"
          << phi::make_ddim(x_dims_array)
          << " ydim:" << phi::make_ddim(y_dims_array);

  CommonGradBroadcastCUDA<T, DX_OP, DY_OP, Tout>(x,
                                                 y,
                                                 out,
                                                 dout,
                                                 dx,
                                                 dy,
                                                 x_dims_array.data(),
                                                 y_dims_array.data(),
                                                 out_dims_array.data(),
                                                 max_dim,
                                                 ctx,
                                                 dx_op,
                                                 dy_op);
}

template <typename T, typename DX_OP, typename DY_OP, typename Tout = T>
void ElemwiseGradComputeWithBroadcast(const GPUContext &ctx,
                                      const DDim &x_dims,
                                      const DDim &y_dims,
                                      const DenseTensor &x,
                                      const DenseTensor &y,
                                      const DenseTensor &out,
                                      const DenseTensor &dout,
                                      int axis,
                                      DenseTensor *dx,
                                      DenseTensor *dy,
                                      DX_OP dx_op,
                                      DY_OP dy_op) {
  bool is_xsize_larger = true;

  int max_dim = x_dims.size();
  if (x_dims.size() < y_dims.size()) {
    is_xsize_larger = false;
    max_dim = y_dims.size();
  }

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  PADDLE_ENFORCE_GE(
      axis,
      0,
      phi::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis,
                    max_dim,
                    phi::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim,
                        axis));

  int pre, n, post, is_run_common_broadcast, axis_trim = 0;
  if (is_xsize_larger) {
    auto y_dims_trimed = funcs::trim_trailing_singular_dims(y_dims);
    axis_trim = (y_dims_trimed.size() == 0) ? x_dims.size() : axis;
    funcs::get_mid_dims(x_dims,
                        y_dims_trimed,
                        axis_trim,
                        &pre,
                        &n,
                        &post,
                        &is_run_common_broadcast);
  } else {
    auto x_dims_trimed = funcs::trim_trailing_singular_dims(x_dims);
    axis_trim = (x_dims_trimed.size() == 0) ? y_dims.size() : axis;
    funcs::get_mid_dims(y_dims,
                        x_dims_trimed,
                        axis_trim,
                        &pre,
                        &n,
                        &post,
                        &is_run_common_broadcast);
  }
  // special case for common backward implementation.
  if (is_run_common_broadcast) {
    CommonElementwiseBroadcastBackward<T, DX_OP, DY_OP, Tout>(
        ctx, x_dims, y_dims, x, y, out, dout, axis, dx, dy, dx_op, dy_op);
    return;
  }
  if (post == 1) {
    ElemwiseGradBroadcast1CUDA(ctx.stream(),
                               x.data<T>(),
                               y.data<T>(),
                               out.data<Tout>(),
                               dout.data<Tout>(),
                               pre,
                               n,
                               is_xsize_larger,
                               dx_op,
                               dy_op,
                               dx == nullptr ? nullptr : ctx.Alloc<T>(dx),
                               dy == nullptr ? nullptr : ctx.Alloc<T>(dy));
  } else {
    ElemwiseGradBroadcast2CUDA(ctx.stream(),
                               x.data<T>(),
                               y.data<T>(),
                               out.data<Tout>(),
                               dout.data<Tout>(),
                               pre,
                               n,
                               post,
                               is_xsize_larger,
                               dx_op,
                               dy_op,
                               dx == nullptr ? nullptr : ctx.Alloc<T>(dx),
                               dy == nullptr ? nullptr : ctx.Alloc<T>(dy));
  }
}

/*
******************************
    Add Grad
******************************
*/

template <typename T>
static __global__ void SimpleElemwiseAddGradCUDAKernel(
    const T *__restrict__ dout, int size, int vec_size, T *dx, T *dy) {
  int tid = BLOCK_ID_X * BLOCK_NUM_X + THREAD_ID_X;
  int stride = GRID_NUM_X * BLOCK_NUM_X;
  int loop = size / vec_size;
  int remainder = size % vec_size;
  const float4 *dout_vec = reinterpret_cast<const float4 *>(dout);
  float4 *dx_vec = reinterpret_cast<float4 *>(dx);
  float4 *dy_vec = reinterpret_cast<float4 *>(dy);
  float4 tmp_loop;

  for (int i = tid; i < loop; i += stride) {
    tmp_loop = dout_vec[i];
    dx_vec[i] = tmp_loop;
    dy_vec[i] = tmp_loop;
  }

  if (tid == loop && remainder != 0) {
    T tmp_rem;
    while (remainder) {
      int idx = size - remainder;
      remainder--;
      tmp_rem = dout[idx];
      dx[idx] = tmp_rem;
      dy[idx] = tmp_rem;
    }
  }
}

template <typename T>
void default_elementwise_add_grad(const GPUContext &ctx,
                                  const DenseTensor &x,
                                  const DenseTensor &y,
                                  const DenseTensor &out,
                                  const DenseTensor &dout,
                                  DenseTensor *dx,
                                  DenseTensor *dy,
                                  int axis = -1) {
  auto *dout_data = dout.data<T>();

  // dx
  if (dx != nullptr) {
    auto *dx_data = dx->mutable_data<T>(ctx.GetPlace());
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        phi::Copy(ctx, dout, ctx.GetPlace(), false, dx);
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->mutable_data<T>(x.dims(), ctx.GetPlace());
      }
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(x.dims(), out.dims(), axis);
      gpuStream_t stream = ctx.stream();
      kernels::TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          ctx, dout, dx, kps::IdentityFunctor<T>(), reduce_dims, stream);
    }
  }
  // dy
  if (dy != nullptr) {
    auto *dy_data = dy->mutable_data<T>(ctx.GetPlace());
    if (dy->dims() == dout.dims()) {
      if (dy_data != dout_data) {
        phi::Copy(ctx, dout, ctx.GetPlace(), false, dy);
      }
    } else {
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(y.dims(), out.dims(), axis);
      gpuStream_t stream = ctx.stream();
      kernels::TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          ctx, dout, dy, kps::IdentityFunctor<T>(), reduce_dims, stream);
    }
  }
}

template <typename T>
void elementwise_add_grad(const GPUContext &ctx,
                          const DenseTensor &x,
                          const DenseTensor &y,
                          const DenseTensor &out,
                          const DenseTensor &dout,
                          DenseTensor *dx,
                          DenseTensor *dy) {
  auto *dx_data = dx->mutable_data<T>(ctx.GetPlace());
  auto *dy_data = dy->mutable_data<T>(ctx.GetPlace());
  auto *dout_data = dout.data<T>();
  if (dx_data == dout_data && dy_data != dout_data) {
    VLOG(4) << "Special case when dx_data is the same as dout_data, "
               "only need copy dout to dy";
    phi::Copy(ctx, dout, ctx.GetPlace(), false, dy);
  } else if (dx_data != dout_data && dy_data == dout_data) {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "only need copy dout to dx";
    phi::Copy(ctx, dout, ctx.GetPlace(), false, dx);
  } else if (dx_data != dout_data && dy_data != dout_data) {
    auto size = x.numel();
    int vec_size = max(static_cast<int>(sizeof(float4) / sizeof(T)), 1);
    dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
    dim3 grid_size =
        dim3(((size + vec_size - 1) / vec_size + PREDEFINED_BLOCK_SIZE - 1) /
                 PREDEFINED_BLOCK_SIZE,
             1);
    SimpleElemwiseAddGradCUDAKernel<
        T><<<grid_size, block_size, 0, ctx.stream()>>>(
        dout.data<T>(),
        size,
        vec_size,
        dx->mutable_data<T>(ctx.GetPlace()),
        dy->mutable_data<T>(ctx.GetPlace()));
  } else {
    VLOG(4) << "Special case when dy_data is the same as dout_data, "
               "and dx_data is the same as dout_data, do not need "
               "any operator";
  }
}

/*
******************************
    Sub Grad
******************************
*/

template <typename T>
static __global__ void SimpleElemwiseSubGradCUDAKernel(const T *dout,
                                                       int64_t size,
                                                       T *dx,
                                                       T *dy) {
  int col = BLOCK_ID_X * BLOCK_NUM_X + THREAD_ID_X;

  while (col < size) {
    if (dx != nullptr) {
      dx[col] = dout[col];
    }
    dy[col] = -dout[col];
    col += BLOCK_NUM_X * GRID_NUM_X;
  }
}

template <typename T>
void default_elementwise_sub_grad(const GPUContext &ctx,
                                  const DenseTensor &x,
                                  const DenseTensor &y,
                                  const DenseTensor &out,
                                  const DenseTensor &dout,
                                  DenseTensor *dx,
                                  DenseTensor *dy,
                                  int axis = -1) {
  auto *dout_data = dout.data<T>();
  // dx
  if (dx != nullptr) {
    auto *dx_data = dx->mutable_data<T>(ctx.GetPlace());
    if (dx->dims() == dout.dims()) {
      if (dx_data != dout_data) {
        phi::Copy(ctx, dout, ctx.GetPlace(), false, dx);
      }
    } else {
      // For inplace strategy, dx will be stored in addr of dout, which makes
      // the result of dy wrong.
      if (dx->IsSharedBufferWith(dout)) {
        dx->clear();
        dx->mutable_data<T>(x.dims(), ctx.GetPlace());
      }
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(x.dims(), out.dims(), axis);
      gpuStream_t stream = ctx.stream();
      kernels::TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          ctx, dout, dx, kps::IdentityFunctor<T>(), reduce_dims, stream);
    }
  }
  // dy
  if (dy != nullptr) {
    auto *dy_data = dy->mutable_data<T>(ctx.GetPlace());
    if (dy->dims() == dout.dims()) {
      if (dy_data != dout_data) {
        dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
        auto size = dy->numel();
        dim3 grid_size =
            dim3((size + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
        SimpleElemwiseSubGradCUDAKernel<
            T><<<grid_size, block_size, 0, ctx.stream()>>>(
            dout.data<T>(), size, nullptr, dy->mutable_data<T>(ctx.GetPlace()));
      }
    } else {
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(y.dims(), out.dims(), axis);
      gpuStream_t stream = ctx.stream();
      kernels::TensorReduceImpl<T, T, kps::AddFunctor, kps::InverseFunctor<T>>(
          ctx, dout, dy, kps::InverseFunctor<T>(), reduce_dims, stream);
    }
  }
}

template <typename T>
void elementwise_sub_grad(const GPUContext &ctx,
                          const DenseTensor &x,
                          const DenseTensor &y,
                          const DenseTensor &out,
                          const DenseTensor &dout,
                          DenseTensor *dx,
                          DenseTensor *dy) {
  dim3 block_size = dim3(PREDEFINED_BLOCK_SIZE, 1);
  auto size = x.numel();
  dim3 grid_size =
      dim3((size + PREDEFINED_BLOCK_SIZE - 1) / PREDEFINED_BLOCK_SIZE, 1);
  SimpleElemwiseSubGradCUDAKernel<
      T><<<grid_size, block_size, 0, ctx.stream()>>>(
      dout.data<T>(),
      size,
      dx->mutable_data<T>(ctx.GetPlace()),
      dy->mutable_data<T>(ctx.GetPlace()));
}

#endif

}  // namespace phi
