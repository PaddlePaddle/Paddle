/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// #define _LOCALDEBUG_

#ifndef _LOCALDEBUG_

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/softmax_impl.cuh"
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#endif

#ifdef _LOCALDEBUG_

#include <iostream>
#include <limits>

namespace platform {

template <typename T>
__forceinline__ __device__ T CudaShuffleXorSync(unsigned mask, T val,
                                                int width = warpSize) {
#if defined(PADDLE_WITH_HIP) || CUDA_VERSION < 9000
  return __shfl_xor(val, width);
#else
  return __shfl_xor_sync(mask, val, width);
#endif
}
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceSum(T *sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T sum_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = sum[i] + sum_val;
    }
  }
}

template <typename T, int BatchSize, int WarpSize>
__device__ __forceinline__ void WarpReduceMax(T *sum) {
#pragma unroll
  for (int offset = WarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < BatchSize; ++i) {
      T max_val = platform::CudaShuffleXorSync(0xFFFFFFFF, sum[i], offset);
      sum[i] = max(sum[i], max_val);
    }
  }
}

#endif

template <typename T, int BlockSize, int NnzBlockMax, bool KpMode = false,
          bool AttnMode = false>
__global__ void BlockSparseSoftmaxForward(T *softmax, const T *src, T scale,
                                          const T *kp_mask, const T *attn_mask,
                                          const int *layout_rowptr,
                                          const int *layout_colindex,
                                          int seq_len) {
  // constants
  const int WarpSize = 32;
  const int BlockSize2 = BlockSize * BlockSize;

  // current thread related info
  const int cur = blockIdx.x * blockDim.y + threadIdx.y;
  const int cur_batch = cur / seq_len;
  const int cur_seq = cur % seq_len;
  const int cur_seqb = cur / BlockSize;
  const int cur_inb = cur % BlockSize;

  // number of nnz block in cur_seqb
  const int cur_nnzb = layout_rowptr[cur_seqb + 1] - layout_rowptr[cur_seqb];

  T srcdata[(BlockSize * NnzBlockMax + WarpSize - 1) / WarpSize];
  T attndata[(BlockSize * NnzBlockMax + WarpSize - 1) / WarpSize];

  // read kp mask
  T datakp_mask = (KpMode == true) ? kp_mask[cur] : 0;

  // read tensor data, attn mask
  if (BlockSize == 1) {  // BlockSize = 1
    const int Iter = (cur_nnzb + WarpSize - 1) / WarpSize;
    const T *srcptr = src + layout_rowptr[cur_seqb];
    const T *attnptr = attn_mask + cur_seqb * seq_len;
    const int *colindex = layout_colindex + layout_rowptr[cur_seqb];
    for (int j = 0; j < Iter; j++) {
      int xidx = j * WarpSize + threadIdx.x;
      int didx = j;
      if (xidx < cur_nnzb) {
        if (AttnMode == true) {
          if (std::abs(attnptr[colindex[xidx]]) <
              std::numeric_limits<T>::epsilon()) {
            srcdata[didx] =
                -std::numeric_limits<T>::infinity() * scale + datakp_mask;
          } else {
            srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
          }
        } else {
          srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
        }
      } else {
        srcdata[didx] = -std::numeric_limits<T>::infinity();
      }
    }
  } else if (BlockSize == 64 || BlockSize == 32) {  // BlockSize = 64, 32

    for (int i = 0; i < cur_nnzb; i++) {
      const T *srcptr = src + layout_rowptr[cur_seqb] * BlockSize2 +
                        i * BlockSize2 + cur_inb * BlockSize;
      const T *attnptr =
          attn_mask + cur_seq * seq_len + layout_colindex[cur_seqb] * BlockSize;
      const int IterPerBlock = BlockSize / WarpSize;
#pragma unroll
      for (int j = 0; j < IterPerBlock; j++) {
        int didx = i * IterPerBlock + j;
        int xidx = threadIdx.x + j * WarpSize;

        if (AttnMode == true) {
          if (std::abs(attnptr[xidx]) < std::numeric_limits<T>::epsilon()) {
            srcdata[didx] =
                -std::numeric_limits<T>::infinity() * scale + datakp_mask;
          } else {
            srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
          }
        } else {
          srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
        }

        // if (threadIdx.x==1 && threadIdx.y==0 && blockIdx.x==63){
        //     printf("%d, %d, %f\n", didx, xidx, srcptr[xidx]);
        // }
      }
    }
  }
  // if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==1){
  //      for (int i=0;i<cur_nnzb * BlockSize / WarpSize;i++){
  //         printf("%f\n", srcdata[i]);
  //      }
  // }

  // max value
  T max_value = srcdata[0];
  const int kIteration = (cur_nnzb * BlockSize + WarpSize - 1) / WarpSize;
#pragma unroll
  for (int it = 1; it < kIteration; ++it) {
    max_value = (max_value > srcdata[it]) ? max_value : srcdata[it];
  }
  WarpReduceMax<T, 1, WarpSize>(&max_value);

  // if (threadIdx.x==0 ){
  //     printf("max: %f, %d \n", max_value, blockIdx.x * 4 + threadIdx.y);
  // }

  // exp sum
  T sum = 0;
#pragma unroll
  for (int it = 0; it < kIteration; ++it) {
    srcdata[it] += std::exp(srcdata[it] - max_value);
    sum += srcdata[it];
  }
  WarpReduceSum<T, 1, WarpSize>(&sum);
  // if (threadIdx.x==0 && threadIdx.y==0){
  //     printf("sum: %f \n", sum);
  // }

  // if (threadIdx.x==0 ){
  // printf("sum: %f, %d \n", sum, blockIdx.x * 4 + threadIdx.y);
  //}

  // compute softmax and write out
  if (BlockSize == 1) {  // BlockSize = 1
    const int Iter = (cur_nnzb + WarpSize - 1) / WarpSize;
    T *softmaxptr = softmax + layout_rowptr[cur_seqb];
    for (int j = 0; j < Iter; j++) {
      int xidx = j * WarpSize + threadIdx.x;
      int didx = j;
      if (xidx < cur_nnzb) {
        softmaxptr[xidx] = srcdata[didx] / sum;
      }
    }
  } else if (BlockSize == 64 || BlockSize == 32) {  // BlockSize = 64, 32
    for (int i = 0; i < cur_nnzb; i++) {
      T *softmaxptr = softmax + layout_rowptr[cur_seqb] * BlockSize2 +
                      i * BlockSize2 + cur_inb * BlockSize;
      const int IterPerBlock = BlockSize / WarpSize;
      for (int j = 0; j < IterPerBlock; j++) {
        int didx = i * IterPerBlock + j;
        int xidx = threadIdx.x + j * WarpSize;
        softmaxptr[xidx] = srcdata[didx] / sum;
        //   if (threadIdx.x==0){
        //       printf("%f\n", xptr[xidx]);
        //   }
      }
    }
  }
}

template <typename T, int BlockSize, int NnzBlockMax>
__global__ void BlockSparseSoftmaxBackward(T *dst, const T *grad, const T *src,
                                           T scale, const int *layout_rowptr,
                                           const int *layout_colindex,
                                           int seq_len) {
  // constants
  const int WarpSize = 32;
  const int BlockSize2 = BlockSize * BlockSize;

  // current thread related info
  const int cur = blockIdx.x * blockDim.y + threadIdx.y;
  const int cur_batch = cur / seq_len;
  const int cur_seq = cur % seq_len;
  const int cur_seqb = cur / BlockSize;
  const int cur_inb = cur % BlockSize;

  // number of nnz block in cur_seqb
  const int cur_nnzb = layout_rowptr[cur_seqb + 1] - layout_rowptr[cur_seqb];

  T srcdata[(BlockSize * NnzBlockMax + WarpSize - 1) / WarpSize];
  T graddata[(BlockSize * NnzBlockMax + WarpSize - 1) / WarpSize];

  // read tensor data, attn mask
  if (BlockSize == 1) {  // BlockSize = 1
    const int Iter = (cur_nnzb + WarpSize - 1) / WarpSize;
    const T *srcptr = src + layout_rowptr[cur_seqb];
    const T *gradptr = grad + layout_rowptr[cur_seqb];
    for (int j = 0; j < Iter; j++) {
      int xidx = j * WarpSize + threadIdx.x;
      int didx = j;
      srcdata[didx] = srcptr[xidx];
      graddata[didx] = gradptr[xidx];
    }

  } else if (BlockSize == 64 || BlockSize == 32) {  // BlockSize = 64, 32
    for (int i = 0; i < cur_nnzb; i++) {
      // BlockSize = 64, 32
      const T *srcptr = src + layout_rowptr[cur_seqb] * BlockSize2 +
                        i * BlockSize2 + cur_inb * BlockSize;
      const T *gradptr = grad + layout_rowptr[cur_seqb] * BlockSize2 +
                         i * BlockSize2 + cur_inb * BlockSize;
      const int IterPerBlock = BlockSize / WarpSize;
#pragma unroll
      for (int j = 0; j < IterPerBlock; j++) {
        int didx = i * IterPerBlock + j;
        int xidx = threadIdx.x + j * WarpSize;
        srcdata[didx] = srcptr[xidx];
        graddata[didx] = gradptr[xidx];
      }
    }
  }

  //   if (threadIdx.x==0){
  //       printf("%f\n", xptr[xidx]);
  //   }

  T sum = 0;
  const int kIteration = cur_nnzb * BlockSize / WarpSize;
#pragma unroll
  for (int it = 0; it < kIteration; ++it) {
    sum += srcdata[it] * graddata[it];
  }
  WarpReduceSum<T, 1, WarpSize>(&sum);

  // compute softmax and write out
  if (BlockSize == 1) {  // BlockSize = 1
    const int Iter = (cur_nnzb + WarpSize - 1) / WarpSize;
    T *dstptr = dst + layout_rowptr[cur_seqb];
    for (int j = 0; j < Iter; j++) {
      int xidx = j * WarpSize + threadIdx.x;
      int didx = j;
      if (xidx < cur_nnzb) {
        dstptr[xidx] = srcdata[didx] / sum;
      }
    }
  } else if (BlockSize == 64 || BlockSize == 32) {  // BlockSize = 64, 32
    for (int i = 0; i < cur_nnzb; i++) {
      // BlockSize = 64, 32
      T *dstptr = dst + layout_rowptr[cur_seqb] * BlockSize2 + i * BlockSize2 +
                  cur_inb * BlockSize;
      const int IterPerBlock = BlockSize / WarpSize;
      for (int j = 0; j < IterPerBlock; j++) {
        int didx = i * IterPerBlock + j;
        int xidx = threadIdx.x + j * WarpSize;
        dstptr[xidx] = scale * srcdata[didx] * (graddata[didx] - sum);
        //   if (threadIdx.x==0){
        //       printf("%f\n", xptr[xidx]);
        //   }
      }
    }
  }
}

#ifndef _LOCALDEBUG_

template <typename T>
class SoftmaxBlockSparseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // input
    auto *x = ctx.Input<Tensor>("X");
    auto *rowptr = ctx.Input<Tensor>("layout_rowptr");
    auto *colidx = ctx.Input<Tensor>("layout_colindex");
    int num_block = rowptr->dims()[0] - 1;

    T scale = ctx.Attr<T>("scale");

    bool kp_mode = ctx.Attr<bool>("kp_mask_mode");
    bool attn_mode = ctx.Attr<bool>("attn_mask_mode");
    auto *kp_mask = (kp_mode == true) ? ctx.Input<Tensor>("kp_mask") : NULL;
    auto *attn_mask =
        (attn_mode == true) ? ctx.Input<Tensor>("attn_mask") : NULL;

    // output
    auto *out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    auto *out_data = out->data<T>();

    //
    const auto x_dims = x->dims();
    // const int rank = x_dims.size();
    // const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);

    const int BlockSize = 64;
    const int NnzBlockMax = 8;

    const int num_batch = x_dims[0];
    const int num_nnz = x_dims[1];
    const int block_size = x_dims[2];

    const int seq_len = num_block * BlockSize;

    const int axis = -1;

    const dim3 blocks(32, 4, 1);
    const int grid = num_batch * seq_len / (32 * 4);

    if ((kp_mode == true) && (attn_mode == true)) {
      BlockSparseSoftmaxForward<
          T, BlockSize, NnzBlockMax, true,
          true><<<grid, blocks, 0, ctx.cuda_device_context().stream()>>>(
          out_data, x->data<T>(), scale, kp_mask->data<T>(),
          attn_mask->data<T>(), rowptr->data<int32_t>(),
          colidx->data<int32_t>(), seq_len);
    } else if ((kp_mode == true) && (attn_mode == false)) {
      BlockSparseSoftmaxForward<
          T, BlockSize, NnzBlockMax, true,
          false><<<grid, blocks, 0, ctx.cuda_device_context().stream()>>>(
          out_data, x->data<T>(), scale, kp_mask->data<T>(), NULL,
          rowptr->data<int32_t>(), colidx->data<int32_t>(), seq_len);
    } else if ((kp_mode == false) && (attn_mode == true)) {
      BlockSparseSoftmaxForward<
          T, BlockSize, NnzBlockMax, false,
          true><<<grid, blocks, 0, ctx.cuda_device_context().stream()>>>(
          out_data, x->data<T>(), scale, NULL, attn_mask->data<T>(),
          rowptr->data<int32_t>(), colidx->data<int32_t>(), seq_len);
    } else {
      BlockSparseSoftmaxForward<
          T, BlockSize, NnzBlockMax, false,
          false><<<grid, blocks, 0, ctx.cuda_device_context().stream()>>>(
          out_data, x->data<T>(), scale, NULL, NULL, rowptr->data<int32_t>(),
          colidx->data<int32_t>(), seq_len);
    }
  }
};

template <typename T>
class SoftmaxBlockSparseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("Out");
    auto *rowptr = ctx.Input<Tensor>("layout_rowptr");
    auto *colidx = ctx.Input<Tensor>("layout_colindex");

    int num_block = rowptr->dims()[0] - 1;

    T scale = ctx.Attr<T>("scale");

    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    auto *dx_data = dx->data<T>();

    const auto x_dims = x->dims();
    // const int rank = x_dims.size();
    // const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);

    const int BlockSize = 64;
    const int NnzBlockMax = 8;

    const int num_batch = x_dims[0];
    const int num_nnz = x_dims[1];
    const int block_size = x_dims[2];
    const int seq_len = num_block * BlockSize;

    const int axis = -1;

    const dim3 blocks(32, 4, 1);
    const int grid = num_batch * seq_len / (32 * 4);
    BlockSparseSoftmaxBackward<
        T, BlockSize,
        NnzBlockMax><<<grid, blocks, 0, ctx.cuda_device_context().stream()>>>(
        dx_data, dout->data<T>(), x->data<T>(), scale, rowptr->data<int32_t>(),
        colidx->data<int32_t>(), seq_len);
  }
};
}
}

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(softmax_blocksparse,
                        ops::SoftmaxBlockSparseKernel<float>);
REGISTER_OP_CUDA_KERNEL(softmax_blocksparse_grad,
                        ops::SoftmaxBlockSparseGradKernel<float>);

#endif

#ifdef _LOCALDEBUG_

int main() {
  const int BlockSize = 64;

  float *a;
  float *out;
  float *d_a, *d_out;

  const int Batch = 2;
  const int MB = 2;
  const int NB = 4;
  const int MBB = MB * Batch;

  const int seqlen = MB * BlockSize;

  int layout[MBB][NB];
  for (int i = 0; i < MBB; i++) {
    for (int j = 0; j < NB; j++) {
      layout[i][j] = 0;
    }
  }
  layout[0][0] = 1;
  layout[0][3] = 1;
  layout[1][0] = 1;
  layout[1][1] = 1;
  layout[2][0] = 1;
  layout[2][2] = 1;
  layout[3][1] = 1;

  int rowptr[MBB + 1];
  rowptr[0] = 0;
  for (int i = 0; i < MBB; i++) {
    rowptr[i + 1] = rowptr[i];
    for (int j = 0; j < NB; j++) {
      rowptr[i + 1] += layout[i][j];
    }
  }

  int colidx[rowptr[MBB]];
  int idx = 0;
  for (int i = 0; i < MBB; i++) {
    for (int j = 0; j < NB; j++) {
      if (layout[i][j] == 1) {
        colidx[idx] = j;
        idx += 1;
      }
    }
  }

  for (int i = 0; i < MBB + 1; i++) {
    std::cout << rowptr[i] << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < rowptr[MBB]; i++) {
    std::cout << colidx[i] << " ";
  }
  std::cout << std::endl;

  int N = rowptr[MBB] * BlockSize * BlockSize;

  a = (float *)malloc(sizeof(float) * N);
  out = (float *)malloc(sizeof(float) * N);
  for (int i = 0; i < N; i++) {
    a[i] = i + 1;
  }

  int *d_rowptr;
  int *d_colidx;
  cudaMalloc((void **)&d_rowptr, sizeof(int) * (MBB + 1));
  cudaMalloc((void **)&d_colidx, sizeof(int) * rowptr[MBB]);
  cudaMemcpy(d_rowptr, rowptr, sizeof(int) * (MBB + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_colidx, colidx, sizeof(int) * rowptr[MBB],
             cudaMemcpyHostToDevice);

  // Allocate device memory
  cudaMalloc((void **)&d_a, sizeof(float) * N);
  cudaMalloc((void **)&d_out, sizeof(float) * N);

  // Transfer data from host to device memory
  cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);

  printf("Kernel start \n");

  dim3 blocks(32, 4, 1);
  int grid = MBB * BlockSize / 4;

  BlockSparseSoftmaxForward<float, BlockSize, 4><<<grid, blocks>>>(
      d_out, d_a, 1.0, NULL, NULL, d_rowptr, d_colidx, seqlen);

  BlockSparseSoftmaxBackward<float, BlockSize, 4><<<grid, blocks>>>(
      NULL, NULL, NULL, 1.0, d_rowptr, d_colidx, seqlen);

  printf("Kernel end \n");

  cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Cleanup after kernel execution
  cudaFree(d_out);
  free(a);
  // free(out);

  return 0;
}

#endif