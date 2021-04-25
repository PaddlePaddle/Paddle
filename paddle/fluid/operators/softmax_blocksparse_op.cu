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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/softmax_impl.cuh"
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, int BlockSize, int NnzBlockMax, bool KpMode = false,
          bool AttnMode = false>
__global__ void BlockSparseSoftmaxForward(T *softmax, const T *src, T scale,
                                          const T *kp_mask, const T *attn_mask,
                                          const int *layout_rowptr,
                                          const int *layout_colidx,
                                          int seq_length) {
  // constants
  const int WarpSize = 32;
  const int BlockSize2 = BlockSize * BlockSize;

  // current thread related info
  const int cur = blockIdx.x * blockDim.y + threadIdx.y;
  const int cur_batch = cur / seq_length;
  const int cur_seq = cur % seq_length;
  const int cur_seqb = cur / BlockSize;
  const int cur_inb = cur % BlockSize;

  // number of nnz block in cur_seqb
  const int cur_nnzb = layout_rowptr[cur_seqb + 1] - layout_rowptr[cur_seqb];

  T srcdata[BlockSize * NnzBlockMax / WarpSize];
  T attndata[BlockSize * NnzBlockMax / WarpSize];

  // read kp mask
  T datakp_mask = (KpMode == true) ? kp_mask[cur] : 0;

  // read tensor data, attn mask
  for (int i = 0; i < cur_nnzb; i++) {
    // BlockSize = 64, 32
    const T *srcptr = src + layout_rowptr[cur_seqb] * BlockSize2 +
                      i * BlockSize2 + cur_inb * BlockSize;
    const T *attnptr =
        attn_mask + cur_seq * seq_length + layout_colidx[cur_seqb] * BlockSize;

    const int IterPerBlock = BlockSize / WarpSize;
#pragma unroll
    for (int j = 0; j < IterPerBlock; j++) {
      int didx = i * IterPerBlock + j;
      int xidx = threadIdx.x + j * WarpSize;

      if (AttnMode == true) {
        if (std::abs(attnptr[xidx]) < std::numeric_limits<T>::epsilon()) {
          srcdata[didx] = -std::numeric_limits<T>::infinity() * scale + datakp_mask;
        } else {
          srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
        }
      } else {
        srcdata[didx] = scale * srcptr[xidx] + datakp_mask;
      }

      // if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0){
      //     printf("%d, %d, %f\n", xidx, didx, srcptr[xidx]);
      // }
    }
  }
  // if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==1){
  //      for (int i=0;i<cur_nnzb * BlockSize / WarpSize;i++){
  //         printf("%f\n", srcdata[i]);
  //      }
  // }

  // max value
  T max_value = srcdata[0];
  const int kIteration = cur_nnzb * BlockSize / WarpSize;
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
  for (int i = 0; i < cur_nnzb; i++) {
    // BlockSize = 64, 32
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

template <typename T, int BlockSize, int NnzBlockMax>
__global__ void BlockSparseSoftmaxBackward(T *dst, T *grad, const T *src, T scale,
                                           const int *layout_rowptr,
                                           const int *layout_colidx,
                                           int seq_length) {
  // constants
  const int WarpSize = 32;
  const int BlockSize2 = BlockSize * BlockSize;

  // current thread related info
  const int cur = blockIdx.x * blockDim.y + threadIdx.y;
  const int cur_batch = cur / seq_length;
  const int cur_seq = cur % seq_length;
  const int cur_seqb = cur / BlockSize;
  const int cur_inb = cur % BlockSize;

  // number of nnz block in cur_seqb
  const int cur_nnzb = layout_rowptr[cur_seqb + 1] - layout_rowptr[cur_seqb];

  T srcdata[BlockSize * NnzBlockMax / WarpSize];
  T graddata[BlockSize * NnzBlockMax / WarpSize];

  // read tensor data, attn mask
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

template <typename T>
class SoftmaxBlockSparseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // input
    auto *x = ctx.Input<Tensor>("X");
    auto *rowptr = ctx.Input<Tensor>("LayOutRowPtr");
    auto *colidx = ctx.Input<Tensor>("LayOutColIndex");
    int num_block = rowptr->dims()[0] - 1;

    // output
    auto *out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    auto *out_data = out->data<T>();

    //
    const auto x_dims = x->dims();
    // const int rank = x_dims.size();
    // const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);

    const int BlockSize = 64;

    const int num_batch = x_dims[0];
    const int num_nnz = x_dims[1];
    const int block_size = x_dims[2];

    const int seqlen = num_block * BlockSize;

    const int axis = -1;

    const dim3 blocks(32, 4, 1);
    const int grid = num_batch * seqlen / (32 * 4);
    BlockSparseSoftmaxForward<T, BlockSize, 4><<<grid, blocks>>>(
        out_data, x->data<T>(), 1.0, NULL, NULL, 
        rowptr->data<int32_t>(), colidx->data<int32_t>(), seqlen);
  }
};

template <typename T>
class SoftmaxBlockSparseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("Out");
    auto *rowptr = ctx.Input<Tensor>("LayOutRowPtr");
    auto *colidx = ctx.Input<Tensor>("LayOutColIndex");
    int num_block = rowptr->dims()[0] - 1;

    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    auto *dx_data = dx->data<T>();

    const auto x_dims = x->dims();
    // const int rank = x_dims.size();
    // const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);

    const int BlockSize = 64;

    const int num_batch = x_dims[0];
    const int num_nnz = x_dims[1];
    const int block_size = x_dims[2];
    const int seqlen = num_block * BlockSize;

    const int axis = -1;

    const dim3 blocks(32, 4, 1);
    const int grid = num_batch * seqlen / (32 * 4);
    BlockSparseSoftmaxBackward<T, BlockSize, 4><<<grid, blocks>>>(
        dx_data, dout->data<T>(), x->data<T>(), 1.0, NULL, NULL, 
        rowptr->data<int32_t>(), colidx->data<int32_t>(), seqlen);
  }
};

}
}

namespace ops = paddle::operators;

#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(sparse_softmax, ops::SoftmaxBlockSparseKernel<float>)
REGISTER_OP_CUDA_KERNEL(sparse_softmax_grad,
                        ops::SoftmaxBlockSparseKernel<float>);
#else
REGISTER_OP_CUDA_KERNEL(sparse_softmax, ops::SoftmaxBlockSparseKernel<float>);
// REGISTER_OP_CUDA_KERNEL(sparse_softmax_grad,
//                    ops::SoftmaxBlockSparseKernel<float>);
#endif
