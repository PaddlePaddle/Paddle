/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/cuda/include/hl_base.h"
#include "paddle/function/RowConvOp.h"

namespace paddle {

template <int BLOCK_H, int BLOCK_W>
__global__ void KeRowConv(real* y,
                          const real* x,
                          const real* w,
                          const int* starts,
                          const int height,
                          const int width,
                          const int numSeq,
                          const int context) {
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int blky = blockDim.y;
  const int gidx = blockIdx.x * blockDim.x;

  __shared__ real sw[BLOCK_H][BLOCK_W];

  for (int i = tidy; i < context; i += blky) {
    sw[i][tidx] = gidx + tidx < width ? w[i * width + gidx + tidx] : 0.0;
  }

  __syncthreads();

  for (int i = 0; i < numSeq; ++i) {
    const int start = starts[i];
    const int end = starts[i + 1];
    const int steps = end - start;
    for (int j = tidy; j < steps; j += blky) {
      real sum = 0;
      int off = (start + j) * width;
      for (int t = 0; t < context; ++t) {
        if ((start + j + t) < end) {
          int xoff = off + t * width;
          real xVal = gidx + tidx < width ? x[xoff + gidx + tidx] : 0.0;
          sum += sw[t][tidx] * xVal;
        }
      }
      if (gidx + tidx < width) {
        y[off + gidx + tidx] += sum;
      }
    }
  }
}

__global__ void KeRowConv2(real* y,
                           const real* x,
                           const real* w,
                           const int* starts,
                           const int height,
                           const int width,
                           const int numSeq,
                           const int context) {
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int blky = blockDim.y;
  const int gidx = blockIdx.x * blockDim.x;

  for (int i = 0; i < numSeq; ++i) {
    const int start = starts[i];
    const int end = starts[i + 1];
    const int steps = end - start;
    for (int j = tidy; j < steps; j += blky) {
      int off = (start + j) * width;
      real sum = 0;
      for (int t = 0; t < context && (start + j + t) < end; ++t) {
        int xoff = off + t * width;
        real xd = gidx + tidx < width ? x[xoff + gidx + tidx] : 0.0;
        real wd = gidx + tidx < width ? w[t * width + gidx + tidx] : 0.0;
        sum += wd * xd;
      }
      if (gidx + tidx < width) {
        y[off + gidx + tidx] += sum;
      }
    }
  }
}

template <>
void RowConv<DEVICE_TYPE_GPU>(GpuMatrix& out,  // NOLINT
                              const GpuMatrix& in,
                              const GpuMatrix& filter,
                              const GpuIVector& seq) {
  const size_t numSeq = seq.getSize() - 1;
  const size_t contextLength = filter.getHeight();
  const size_t height = in.getHeight();
  const size_t width = in.getWidth();

  real* y = out.getData();
  const real* x = in.getData();
  const real* w = filter.getData();
  const int* starts = seq.getData();

  dim3 dimBlock(32, 32);
  dim3 dimGrid(DIVUP(width, dimBlock.x), 1);

  if (contextLength <= 32) {
    KeRowConv<32, 32><<<dimGrid, dimBlock, 0, STREAM_DEFAULT>>>(
        y, x, w, starts, height, width, numSeq, contextLength);
  } else {
    KeRowConv2<<<dimGrid, dimBlock, 0, STREAM_DEFAULT>>>(
        y, x, w, starts, height, width, numSeq, contextLength);
  }
  CHECK_SYNC("RowConv");
}

template <int BLOCK_H, int BLOCK_W, int CONTEXT>
__global__ void KeRowConvBwWeight(real* dw,
                                  const real* x,
                                  const real* dy,
                                  const int* starts,
                                  const int height,
                                  const int width,
                                  const int numSeq,
                                  const int context) {
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int blky = blockDim.y;
  const int gidx = blockIdx.x * blockDim.x;

  __shared__ real sh_x[BLOCK_W][BLOCK_H];
  __shared__ real sh_dy[BLOCK_W][BLOCK_H + CONTEXT - 1];
  __shared__ real sh_dw[CONTEXT][BLOCK_W];

  if (tidy < context) {
    sh_dw[tidy][tidx] = 0.0;
  }
  __syncthreads();

  // NOTE(zcd): temporary solution
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);

  for (int i = 0; i < numSeq; ++i) {
    const int start = starts[i];
    const int end = starts[i + 1];
    const int steps = end - start;
    const int size = ((steps + BLOCK_H - 1) / BLOCK_H) * BLOCK_H;
    for (int j = tidy; j < size; j += BLOCK_H) {
      int xoff = gidx + tidx;
      int yoff = start + j;

      // transpose
      sh_x[tidx][tidy] =
          (xoff < width && yoff < end) ? x[yoff * width + xoff] : 0.0;
      sh_dy[tidx][tidy + context - 1] =
          (xoff < width && yoff < end) ? dy[yoff * width + xoff] : 0.0;
      __syncthreads();
      if (tidy < (context - 1)) {
        yoff = yoff - context + 1;
        sh_dy[tidx][tidy] =
            (xoff < width && yoff >= start) ? dy[yoff * width + xoff] : 0.0;
      }
      __syncthreads();

      for (int t = 0; t < context; t++) {
        real val = sh_x[tidy][tidx] * sh_dy[tidy][tidx + context - 1 - t];
        __syncthreads();
        // warp size and blockDim.x is 32.

        for (int offset = 16; offset > 0; offset /= 2)
          val += __shfl_down_sync(mask, val, offset);

        __syncthreads();
        if (tidx == 0) {
          sh_dw[t][tidy] += val;
        }
        __syncthreads();
      }
    }
  }

  for (int t = tidy; (t < context) && ((gidx + tidx) < width); t += blky) {
    dw[t * width + gidx + tidx] += sh_dw[t][tidx];
  }
}

template <int BLOCK_H, int BLOCK_W>
__global__ void KeRowConvBwWeight2(real* dw,
                                   const real* x,
                                   const real* dy,
                                   const int* starts,
                                   const int height,
                                   const int width,
                                   const int numSeq,
                                   const int context) {
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int gidx = blockIdx.x * blockDim.x;

  __shared__ real sh_x[BLOCK_H][BLOCK_W];
  __shared__ real sh_dy[BLOCK_H][BLOCK_W];

  // NOTE(zcd): temporary solution
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);

  for (int i = 0; i < numSeq; ++i) {
    const int start = starts[i];
    const int end = starts[i + 1];
    const int steps = end - start;

    const int size = ((steps + BLOCK_H - 1) / BLOCK_H) * BLOCK_H;
    for (int j = tidy; j < size; j += BLOCK_H) {
      int xoff = gidx + tidx;
      int yoff = start + j;

      // transpose
      sh_x[tidx][tidy] =
          (xoff < width && yoff < end) ? x[yoff * width + xoff] : 0.0;
      __syncthreads();

      for (int t = 0; t < context; t++) {
        sh_dy[tidx][tidy] =
            (xoff < width && (yoff - t) >= start && yoff - t < end)
                ? dy[(yoff - t) * width + xoff]
                : 0.0;
        __syncthreads();

        real val = sh_x[tidy][tidx] * sh_dy[tidy][tidx];
        __syncthreads();
        // warp size and blockDim.x is 32.
        for (int offset = 16; offset > 0; offset /= 2)
          val += __shfl_down_sync(mask, val, offset);

        __syncthreads();

        if (tidx == 0 && (gidx + tidy) < width) {
          dw[t * width + gidx + tidy] += val;
        }
      }
    }
  }
}

template <int BLOCK_H, int BLOCK_W>
__global__ void KeRowConvBwData(real* dx,
                                const real* w,
                                const real* dy,
                                const int* starts,
                                const int height,
                                const int width,
                                const int numSeq,
                                const int context) {
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int blky = blockDim.y;
  const int gidx = blockIdx.x * blockDim.x;

  __shared__ real sw[BLOCK_H][BLOCK_W];

  for (int i = tidy; i < context; i += blky) {
    sw[i][tidx] = gidx + tidx < width ? w[i * width + gidx + tidx] : 0.0;
  }

  __syncthreads();

  for (int i = 0; i < numSeq; ++i) {
    const int start = starts[i];
    const int end = starts[i + 1];
    const int steps = end - start;
    for (int j = tidy; j < steps; j += blky) {
      real sum = 0;
      int off = (start + j) * width;
      for (int t = 0; t < context && (j - t) >= 0; ++t) {
        int dyOff = off - t * width;
        real dyVal = gidx + tidx < width ? dy[dyOff + gidx + tidx] : 0.0;
        sum += sw[t][tidx] * dyVal;
      }
      if (gidx + tidx < width) {
        dx[off + gidx + tidx] += sum;
      }
    }
  }
}

__global__ void KeRowConvBwData2(real* dx,
                                 const real* w,
                                 const real* dy,
                                 const int* starts,
                                 const int height,
                                 const int width,
                                 const int numSeq,
                                 const int context) {
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int blky = blockDim.y;
  const int gidx = blockIdx.x * blockDim.x;

  for (int i = 0; i < numSeq; ++i) {
    const int start = starts[i];
    const int end = starts[i + 1];
    const int steps = end - start;
    for (int j = tidy; j < steps; j += blky) {
      real sum = 0;
      int off = (start + j) * width;
      for (int t = 0; t < context && (j - t) >= 0; ++t) {
        int dyOff = off - t * width;
        real dyVal = gidx + tidx < width ? dy[dyOff + gidx + tidx] : 0.0;
        real wVal = gidx + tidx < width ? w[t * width + gidx + tidx] : 0.0;
        sum += wVal * dyVal;
      }
      if (gidx + tidx < width) {
        dx[off + gidx + tidx] += sum;
      }
    }
  }
}

template <>
void RowConvGrad<DEVICE_TYPE_GPU>(const GpuMatrix& outG,
                                  const GpuMatrix& in,
                                  const GpuMatrix& filter,
                                  GpuMatrix& inG,      // NOLINT
                                  GpuMatrix& filterG,  // NOLINT
                                  const GpuIVector& seq) {
  const size_t numSeq = seq.getSize() - 1;
  const size_t contextLength = filter.getHeight();
  const size_t height = in.getHeight();
  const size_t width = in.getWidth();

  const real* dy = outG.getData();
  const real* x = in.getData();
  const real* w = filter.getData();
  const int* starts = seq.getData();

  if (filterG) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(DIVUP(width, dimBlock.x), 1);
    real* dw = filterG.getData();
    if (contextLength <= 32) {
      KeRowConvBwWeight<32, 32, 32><<<dimGrid, dimBlock, 0, STREAM_DEFAULT>>>(
          dw, x, dy, starts, height, width, numSeq, contextLength);
    } else {
      KeRowConvBwWeight2<32, 32><<<dimGrid, dimBlock, 0, STREAM_DEFAULT>>>(
          dw, x, dy, starts, height, width, numSeq, contextLength);
    }
  }

  if (inG) {
    real* dx = inG.getData();
    dim3 dimBlock2(32, 32);
    dim3 dimGrid2(DIVUP(width, dimBlock2.x), 1);
    if (contextLength <= 64) {
      KeRowConvBwData<32, 64><<<dimGrid2, dimBlock2, 0, STREAM_DEFAULT>>>(
          dx, w, dy, starts, height, width, numSeq, contextLength);
    } else {
      KeRowConvBwData2<<<dimGrid2, dimBlock2, 0, STREAM_DEFAULT>>>(
          dx, w, dy, starts, height, width, numSeq, contextLength);
    }
  }

  CHECK_SYNC("RowConvGrad");
}

}  // namespace paddle
