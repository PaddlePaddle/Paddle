// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "glog/logging.h"

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/dropout_impl.cu.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/funcs/transpose_function.cu.h"
#include "paddle/phi/kernels/fusion/gpu/fused_softmax_mask_utils.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace phi {
namespace fusion {

class AttnDropoutParam {
 public:
  AttnDropoutParam() {
    is_test_ = false;
    dropout_implementation_ = "downgrade_in_infer";
    dropout_prob_ = 0.5;
    is_upscale_in_train_ = false;
    is_fix_seed_ = false;
    seed_val_ = 0;
    seed_ = nullptr;
  }
  AttnDropoutParam(bool is_test,
                   const std::string dropout_implementation,
                   float dropout_prob,
                   bool is_upscale_in_train,
                   bool is_fix_seed,
                   int seed_val,
                   const phi::DenseTensor* seed) {
    is_test_ = is_test;
    dropout_implementation_ = dropout_implementation;
    dropout_prob_ = dropout_prob;
    is_upscale_in_train_ = is_upscale_in_train;
    is_fix_seed_ = is_fix_seed;
    seed_val_ = seed_val;
    seed_ = seed;
  }
  bool is_test_;
  std::string dropout_implementation_;
  float dropout_prob_;
  bool is_upscale_in_train_;
  bool is_fix_seed_;
  int seed_val_;
  const phi::DenseTensor* seed_;
};

template <typename T, int VecSize>
__global__ void TransposeRemovingPadding(const T* input_data,
                                         T* output_data,
                                         const int batch_size,
                                         const int num_head,
                                         const int seq_len,
                                         const int head_dim,
                                         const int token_num,
                                         const int elem_cnt,
                                         const int* padding_offset) {
  // transpose and remove padding
  // [batch_size, num_head, seq_len, head_dim] -> [token_num, num_head,
  // head_dim]
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int dim_embed = num_head * head_dim;
  using LoadT = phi::AlignedVector<T, VecSize>;
  LoadT src_vec;

  for (int32_t linear_index = idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    const int token_idx = linear_index / dim_embed;
    const int ori_token_idx =
        token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
    const int ori_batch_id = ori_token_idx / seq_len;
    const int ori_seq_id = ori_token_idx % seq_len;
    const int ori_head_id = (linear_index % dim_embed) / head_dim;
    const int ori_head_lane = (linear_index % dim_embed) % head_dim;
    const int ori_idx = ori_batch_id * num_head * seq_len * head_dim +
                        ori_head_id * seq_len * head_dim +
                        ori_seq_id * head_dim + ori_head_lane;
    phi::Load<T, VecSize>(&input_data[ori_idx], &src_vec);
    phi::Store<T, VecSize>(src_vec, &output_data[linear_index]);
  }
}

template <typename T>
void InvokeTransposeRemovePadding(const phi::GPUContext& dev_ctx,
                                  const T* input_data,
                                  T* output_data,
                                  const int batch_size,
                                  const int num_head,
                                  const int seq_len,
                                  const int head_dim,
                                  const int token_num,
                                  const int* padding_offset) {
  // [batch_size, num_head, seq_len, head_dim] -> [token_num, num_head,
  // head_dim]
  constexpr int VEC_16B = 16;
  const int elem_cnt = token_num * num_head * head_dim;
  constexpr int PackSize = VEC_16B / sizeof(T);
  PADDLE_ENFORCE_EQ(
      head_dim % PackSize,
      0,
      errors::PreconditionNotMet(
          "dim_head=%d must be divisible by vec_size=%d", head_dim, PackSize));
  const int32_t pack_num = elem_cnt / PackSize;
  const int32_t block_size = 128;
  int32_t grid_size = (pack_num + block_size - 1) / block_size;
  TransposeRemovingPadding<T, PackSize>
      <<<grid_size, block_size, 0, dev_ctx.stream()>>>(input_data,
                                                       output_data,
                                                       batch_size,
                                                       num_head,
                                                       seq_len,
                                                       head_dim,
                                                       token_num,
                                                       elem_cnt,
                                                       padding_offset);
}

template <typename T>
class FMHARef {
 public:
  FMHARef(const phi::GPUContext& dev_ctx,
          int64_t batch_size,
          int64_t seq_len,
          int64_t num_head,
          int64_t head_dim,
          AttnDropoutParam param)
      : dev_ctx_(dev_ctx),
        batch_size_(batch_size),
        seq_len_(seq_len),
        num_head_(num_head),
        head_dim_(head_dim),
        dropout_param_(param) {}

  ~FMHARef() {}

  void ComputeForward(const phi::DenseTensor& qkv_input_tensor,
                      const phi::DenseTensor* cache_kv_tensor,
                      const phi::DenseTensor* src_mask_tensor,
                      phi::DenseTensor* transpose_2_out_tensor,
                      phi::DenseTensor* cache_kv_out_tensor,
                      phi::DenseTensor* qk_out_tensor,
                      phi::DenseTensor* src_mask_out_tensor,
                      phi::DenseTensor* softmax_out_tensor,
                      phi::DenseTensor* dropout_mask_out_tensor,
                      phi::DenseTensor* dropout_out_tensor,
                      phi::DenseTensor* qktv_out_tensor,
                      phi::DenseTensor* fmha_out_tensor) {
    // input shape: [bs, seq_len, 3, num_head, head_dim]
    // transpose with perm [2, 0, 3, 1, 4],
    // output_shape: [3, bs, num_head, seq_len, head_dim]
    std::vector<int> perm_1 = {2, 0, 3, 1, 4};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, qkv_input_tensor, perm_1, transpose_2_out_tensor);
    T* qkv_data = transpose_2_out_tensor->data<T>();
    T* qk_out_data = qk_out_tensor->data<T>();
    T* qktv_out_data = qktv_out_tensor->data<T>();
    T* softmax_out_data = softmax_out_tensor->data<T>();
    T* fmha_out_data = fmha_out_tensor->data<T>();

    auto out_seq_len = seq_len_;
    if (cache_kv_tensor) {
      // kv [2, bs, num_head, seq_len, head_dim]
      auto kv_tensor = transpose_2_out_tensor->Slice(1, 3);
      phi::funcs::ConcatFunctor<phi::GPUContext, T> concat;
      // out [2, bs, num_head, cache_seq_len + seq_len, head_dim]
      concat(dev_ctx_, {*cache_kv_tensor, kv_tensor}, 3, cache_kv_out_tensor);
      out_seq_len = cache_kv_out_tensor->dims()[3];
    }

    int64_t q_size = batch_size_ * seq_len_ * num_head_ * head_dim_;
    T* q_ptr = qkv_data;
    T* k_ptr = nullptr;
    T* v_ptr = nullptr;

    if (cache_kv_tensor) {
      int64_t k_size = cache_kv_out_tensor->numel() / 2;
      k_ptr = cache_kv_out_tensor->data<T>();
      v_ptr = k_ptr + k_size;
    } else {
      int64_t k_size = q_size;
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + k_size;
    }

    {
      // NOTE(wangxi): We scale Q with 1/sqrt(Dh) before QK^T, because for
      // float16 calculation, INF may appear in QK^T if we do not scale before.
      float alpha = 1.0 / sqrt(head_dim_);
      auto q_tensor = transpose_2_out_tensor->Slice(0, 1);
      auto functor = phi::funcs::ScaleFunctor<T>(alpha);
      std::vector<const phi::DenseTensor*> ins = {&q_tensor};
      std::vector<phi::DenseTensor*> outs = {&q_tensor};
      phi::funcs::ElementwiseKernel<T>(dev_ctx_, ins, &outs, functor);
    }

    // q*k^t, batched_gemm
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasTrans;
    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx_);
    int gemm_batch_size = batch_size_ * num_head_;
    int gemm_m = seq_len_;
    int gemm_n = out_seq_len;
    int gemm_k = head_dim_;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    int64_t stride_a = gemm_m * gemm_k;
    int64_t stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA,
                     transB,
                     gemm_m,
                     gemm_n,
                     gemm_k,
                     alpha,
                     q_ptr,
                     k_ptr,
                     beta,
                     qk_out_data,
                     gemm_batch_size,
                     stride_a,
                     stride_b);
    int softmax_axis = -1;
    if (src_mask_tensor != nullptr) {
      if (src_mask_out_tensor == nullptr && seq_len_ == out_seq_len) {
        phi::fusion::LaunchFusedSoftmaxMaskKernel<T>(qk_out_data,
                                                     src_mask_tensor->data<T>(),
                                                     softmax_out_data,
                                                     batch_size_,
                                                     num_head_,
                                                     seq_len_,
                                                     dev_ctx_.stream());
      } else {
        std::vector<const phi::DenseTensor*> ins;
        std::vector<phi::DenseTensor*> outs;
        ins.emplace_back(qk_out_tensor);
        ins.emplace_back(src_mask_tensor);
        outs.emplace_back(src_mask_out_tensor);
        int elewise_add_axis = -1;
        phi::funcs::BroadcastKernel<T>(dev_ctx_,
                                       ins,
                                       &outs,
                                       phi::funcs::AddFunctor<T>(),
                                       elewise_add_axis);

        phi::SoftmaxForwardCUDAKernelDriver<T>(
            dev_ctx_, *src_mask_out_tensor, softmax_axis, softmax_out_tensor);
      }
    } else {
      phi::SoftmaxForwardCUDAKernelDriver<T>(
          dev_ctx_, *qk_out_tensor, softmax_axis, softmax_out_tensor);
    }

    transB = CblasNoTrans;
    gemm_m = seq_len_;
    gemm_n = head_dim_;
    gemm_k = out_seq_len;
    alpha = static_cast<T>(1.0);
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;

    if (dropout_param_.dropout_prob_) {
      phi::funcs::DropoutFwGPUKernelDriver<T>(
          static_cast<const phi::GPUContext&>(dev_ctx_),
          dropout_param_.is_test_,
          dropout_param_.dropout_prob_,
          dropout_param_.is_upscale_in_train_,
          dropout_param_.is_fix_seed_,
          dropout_param_.seed_val_,
          static_cast<const phi::DenseTensor&>(*softmax_out_tensor),
          dropout_param_.seed_,
          dropout_mask_out_tensor,
          dropout_out_tensor,
          false);
      T* dropout_out_data = dropout_out_tensor->data<T>();
      blas.BatchedGEMM(transA,
                       transB,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       alpha,
                       dropout_out_data,
                       v_ptr,
                       beta,
                       qktv_out_data,
                       gemm_batch_size,
                       stride_a,
                       stride_b);
    } else {
      // softmax_out * v, batched_gemm
      // output shape: [batch_size, num_heads, seq_len, head_dim]
      blas.BatchedGEMM(transA,
                       transB,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       alpha,
                       softmax_out_data,
                       v_ptr,
                       beta,
                       qktv_out_data,
                       gemm_batch_size,
                       stride_a,
                       stride_b);
    }
    // transpose: [0, 2, 1, 3]
    // output shape: [batch_size, seq_len, num_heads, head_dim]
    std::vector<int> perm_3 = {0, 2, 1, 3};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, *qktv_out_tensor, perm_3, fmha_out_tensor);
  }

  void ComputeForwardWithoutTranspose(
      const phi::DenseTensor* cache_kv_tensor,
      const phi::DenseTensor* src_mask_tensor,
      const phi::DenseTensor* padding_offset_tensor,
      phi::DenseTensor* q_transpose_out_tensor,
      phi::DenseTensor* kv_transpose_out_tensor,
      phi::DenseTensor* cache_kv_out_tensor,
      phi::DenseTensor* qk_out_tensor,
      phi::DenseTensor* src_mask_out_tensor,
      phi::DenseTensor* softmax_out_tensor,
      phi::DenseTensor* dropout_mask_out_tensor,
      phi::DenseTensor* dropout_out_tensor,
      phi::DenseTensor* qktv_out_tensor,
      phi::DenseTensor* fmha_out_tensor,
      const int token_num) {
    // input shape: [bs, seq_len, 3, num_head, head_dim]
    // transpose with perm [2, 0, 3, 1, 4],
    // output_shape: [3, bs, num_head, seq_len, head_dim]
    T* qk_out_data = qk_out_tensor->data<T>();
    T* qktv_out_data = qktv_out_tensor->data<T>();
    T* softmax_out_data = softmax_out_tensor->data<T>();
    T* dropout_out_data = dropout_out_tensor->data<T>();
    T* fmha_out_data = fmha_out_tensor->data<T>();

    auto out_seq_len = seq_len_;
    if (cache_kv_tensor) {
      // kv [2, bs, num_head, seq_len, head_dim]
      phi::funcs::ConcatFunctor<phi::GPUContext, T> concat;
      // out [2, bs, num_head, cache_seq_len + seq_len, head_dim]
      concat(dev_ctx_,
             {*cache_kv_tensor, *kv_transpose_out_tensor},
             3,
             cache_kv_out_tensor);
      out_seq_len = cache_kv_out_tensor->dims()[3];
    }

    int64_t q_size = batch_size_ * seq_len_ * num_head_ * head_dim_;
    T* q_ptr = q_transpose_out_tensor->data<T>();
    T* k_ptr = nullptr;
    T* v_ptr = nullptr;

    if (cache_kv_tensor) {
      int64_t k_size = cache_kv_out_tensor->numel() / 2;
      k_ptr = cache_kv_out_tensor->data<T>();
      v_ptr = k_ptr + k_size;
    } else {
      int64_t k_size = q_size;
      k_ptr = kv_transpose_out_tensor->data<T>();
      v_ptr = k_ptr + k_size;
    }

    {
      // NOTE(wangxi): We scale Q with 1/sqrt(Dh) before QK^T, because for
      // float16 calculation, INF may appear in QK^T if we do not scale before.
      float alpha = 1.0 / sqrt(head_dim_);
      auto functor = phi::funcs::ScaleFunctor<T>(alpha);
      std::vector<const phi::DenseTensor*> ins = {q_transpose_out_tensor};
      std::vector<phi::DenseTensor*> outs = {q_transpose_out_tensor};
      phi::funcs::ElementwiseKernel<T>(dev_ctx_, ins, &outs, functor);
    }

    // q*k^t, batched_gemm
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasTrans;
    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx_);
    int gemm_batch_size = batch_size_ * num_head_;
    int gemm_m = seq_len_;
    int gemm_n = out_seq_len;
    int gemm_k = head_dim_;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    int64_t stride_a = gemm_m * gemm_k;
    int64_t stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA,
                     transB,
                     gemm_m,
                     gemm_n,
                     gemm_k,
                     alpha,
                     q_ptr,
                     k_ptr,
                     beta,
                     qk_out_data,
                     gemm_batch_size,
                     stride_a,
                     stride_b);
    int softmax_axis = -1;
    if (src_mask_tensor != nullptr) {
      if (src_mask_out_tensor == nullptr && seq_len_ == out_seq_len) {
        phi::fusion::LaunchFusedSoftmaxMaskKernel<T>(qk_out_data,
                                                     src_mask_tensor->data<T>(),
                                                     softmax_out_data,
                                                     batch_size_,
                                                     num_head_,
                                                     seq_len_,
                                                     dev_ctx_.stream());
      } else {
        std::vector<const phi::DenseTensor*> ins;
        std::vector<phi::DenseTensor*> outs;
        ins.emplace_back(qk_out_tensor);
        ins.emplace_back(src_mask_tensor);
        outs.emplace_back(src_mask_out_tensor);
        int elewise_add_axis = -1;
        phi::funcs::BroadcastKernel<T>(dev_ctx_,
                                       ins,
                                       &outs,
                                       phi::funcs::AddFunctor<T>(),
                                       elewise_add_axis);

        phi::SoftmaxForwardCUDAKernelDriver<T>(
            dev_ctx_, *src_mask_out_tensor, softmax_axis, softmax_out_tensor);
      }
    } else {
      phi::SoftmaxForwardCUDAKernelDriver<T>(
          dev_ctx_, *qk_out_tensor, softmax_axis, softmax_out_tensor);
    }

    transB = CblasNoTrans;
    gemm_m = seq_len_;
    gemm_n = head_dim_;
    gemm_k = out_seq_len;
    alpha = static_cast<T>(1.0);
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;

    if (dropout_param_.dropout_prob_) {
      phi::funcs::DropoutFwGPUKernelDriver<T>(
          static_cast<const phi::GPUContext&>(dev_ctx_),
          dropout_param_.is_test_,
          dropout_param_.dropout_prob_,
          dropout_param_.is_upscale_in_train_,
          dropout_param_.is_fix_seed_,
          dropout_param_.seed_val_,
          static_cast<const phi::DenseTensor&>(*softmax_out_tensor),
          dropout_param_.seed_,
          dropout_mask_out_tensor,
          dropout_out_tensor,
          false);
      blas.BatchedGEMM(transA,
                       transB,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       alpha,
                       dropout_out_data,
                       v_ptr,
                       beta,
                       qktv_out_data,
                       gemm_batch_size,
                       stride_a,
                       stride_b);
    } else {
      // softmax_out * v, batched_gemm
      // output shape: [batch_size, num_heads, seq_len, head_dim]
      blas.BatchedGEMM(transA,
                       transB,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       alpha,
                       softmax_out_data,
                       v_ptr,
                       beta,
                       qktv_out_data,
                       gemm_batch_size,
                       stride_a,
                       stride_b);
    }
    // transpose: [0, 2, 1, 3]
    // output shape: [batch_size, seq_len, num_heads, head_dim]
    if (!padding_offset_tensor) {
      std::vector<int> perm_3 = {0, 2, 1, 3};
      phi::funcs::TransposeGPUKernelDriver<T>(
          dev_ctx_, *qktv_out_tensor, perm_3, fmha_out_tensor);
    } else {
      InvokeTransposeRemovePadding<T>(dev_ctx_,
                                      qktv_out_data,
                                      fmha_out_data,
                                      batch_size_,
                                      num_head_,
                                      seq_len_,
                                      head_dim_,
                                      token_num,
                                      padding_offset_tensor->data<int>());
    }
  }

  void ComputeBackward(const phi::DenseTensor& transpose_2_out_tensor,
                       const phi::DenseTensor* src_mask_tensor,
                       const phi::DenseTensor& softmax_out_tensor,
                       const phi::DenseTensor& dropout_mask_out_tensor,
                       const phi::DenseTensor& dropout_out_tensor,
                       const phi::DenseTensor& qk_out_tensor,
                       const phi::DenseTensor& src_mask_out_tensor,
                       const phi::DenseTensor& fmha_out_grad_tensor,
                       phi::DenseTensor* qktv_out_grad_tensor,
                       phi::DenseTensor* dropout_out_grad_tensor,
                       phi::DenseTensor* softmax_out_grad_tensor,
                       phi::DenseTensor* src_mask_out_grad_tensor,
                       phi::DenseTensor* qk_out_grad_tensor,
                       phi::DenseTensor* transpose_2_out_grad_tensor,
                       phi::DenseTensor* src_mask_grad_tensor,
                       phi::DenseTensor* qkv_input_grad_tensor) {
    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx_);
    int q_size = batch_size_ * seq_len_ * num_head_ * head_dim_;
    int k_size = q_size;
    int softmax_axis = -1;

    T* qkv_grad_data = transpose_2_out_grad_tensor->data<T>();
    T* q_grad_ptr = qkv_grad_data;
    T* k_grad_ptr = q_grad_ptr + q_size;
    T* v_grad_ptr = k_grad_ptr + k_size;
    const T* qkv_data = transpose_2_out_tensor.data<T>();
    const T* q_ptr = qkv_data;
    const T* k_ptr = q_ptr + q_size;
    const T* v_ptr = k_ptr + k_size;

    const T* softmax_out_data = softmax_out_tensor.data<T>();
    T* softmax_out_grad_data = softmax_out_grad_tensor->data<T>();
    T* qktv_out_grad_data = qktv_out_grad_tensor->data<T>();

    // transpose bw
    std::vector<int> perm_3 = {0, 2, 1, 3};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, fmha_out_grad_tensor, perm_3, qktv_out_grad_tensor);

    // recall batchedgemm(nn) fw: softmax_out_data(x) * v_ptr(y) =
    // qktv_out_data(out)
    CBLAS_TRANSPOSE transA = CblasTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    int gemm_batch_size = batch_size_ * num_head_;
    int gemm_m = seq_len_;
    int gemm_n = head_dim_;
    int gemm_k = seq_len_;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    int64_t stride_a = gemm_m * gemm_k;
    int64_t stride_b = gemm_k * gemm_n;
    // bw: dy = x^t * dout
    if (dropout_param_.dropout_prob_) {
      const T* dropout_out_data = dropout_out_tensor.data<T>();
      blas.BatchedGEMM(transA,
                       transB,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       alpha,
                       dropout_out_data,
                       qktv_out_grad_data,
                       beta,
                       v_grad_ptr,
                       gemm_batch_size,
                       stride_a,
                       stride_b);
    } else {
      blas.BatchedGEMM(transA,
                       transB,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       alpha,
                       softmax_out_data,
                       qktv_out_grad_data,
                       beta,
                       v_grad_ptr,
                       gemm_batch_size,
                       stride_a,
                       stride_b);
    }
    // bw: dx = dout * y^t
    transA = CblasNoTrans;
    transB = CblasTrans;
    gemm_m = seq_len_;
    gemm_n = seq_len_;
    gemm_k = head_dim_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    if (dropout_param_.dropout_prob_) {
      T* dropout_out_grad_data = dropout_out_grad_tensor->data<T>();
      blas.BatchedGEMM(transA,
                       transB,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       alpha,
                       qktv_out_grad_data,
                       v_ptr,
                       beta,
                       dropout_out_grad_data,
                       gemm_batch_size,
                       stride_a,
                       stride_b);
    } else {
      blas.BatchedGEMM(transA,
                       transB,
                       gemm_m,
                       gemm_n,
                       gemm_k,
                       alpha,
                       qktv_out_grad_data,
                       v_ptr,
                       beta,
                       softmax_out_grad_data,
                       gemm_batch_size,
                       stride_a,
                       stride_b);
    }
    // dropout bw
    if (dropout_param_.dropout_prob_) {
      phi::funcs::DropoutGradGPUKernelDriver<T>(
          static_cast<const phi::GPUContext&>(dev_ctx_),
          false,
          dropout_param_.dropout_prob_,
          dropout_param_.is_upscale_in_train_,
          static_cast<const phi::DenseTensor&>(*dropout_out_grad_tensor),
          dropout_mask_out_tensor,
          softmax_out_grad_tensor,
          false);
    }

    if (src_mask_tensor != nullptr) {
      phi::SoftmaxBackwardCUDAKernelDriver<T>(dev_ctx_,
                                              softmax_out_tensor,
                                              *softmax_out_grad_tensor,
                                              softmax_axis,
                                              src_mask_out_grad_tensor);
      // recall LaunchElementwiseCudaKernel fw:  src_mask_out = qk_out +
      // src_mask
      // Special case when dy is not needed and dx doesn't reduce
      if (qk_out_grad_tensor != nullptr && src_mask_grad_tensor == nullptr &&
          qk_out_tensor.dims() == src_mask_out_tensor.dims()) {
        VLOG(4) << "Special case when dy is not needed and dx doesn't "
                   "reduce";
        phi::Copy(dev_ctx_,
                  *src_mask_out_grad_tensor,
                  dev_ctx_.GetPlace(),
                  false,
                  qk_out_grad_tensor);
      } else {
        PADDLE_THROW(errors::InvalidArgument(
            "Only used for the backward elementwise_add op when"
            "dy is not needed and dx is not reduce"));
        return;
      }

    } else {
      phi::SoftmaxBackwardCUDAKernelDriver<T>(dev_ctx_,
                                              softmax_out_tensor,
                                              *softmax_out_grad_tensor,
                                              softmax_axis,
                                              qk_out_grad_tensor);
    }

    T* qk_out_grad_data = qk_out_grad_tensor->data<T>();
    // NOTE(wangxi): For we scale Q with 1/sqrt(Dh) in forward, so we set
    //   alpha = 1.0 in backward.
    alpha = static_cast<T>(1.0);
    // recall batchedgemm(nt) fw:  q_ptr * (k_ptr)^t = qk_out
    // bw: dy (seq_len * head_dim) = (dout)^t * x
    transA = CblasTrans;
    transB = CblasNoTrans;
    gemm_m = seq_len_;
    gemm_n = head_dim_;
    gemm_k = seq_len_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA,
                     transB,
                     gemm_m,
                     gemm_n,
                     gemm_k,
                     alpha,
                     qk_out_grad_data,
                     q_ptr,
                     beta,
                     k_grad_ptr,
                     gemm_batch_size,
                     stride_a,
                     stride_b);
    // dx (seq_len * head_dim) = dout * y
    alpha = static_cast<T>(1.0 / sqrt(head_dim_));
    transA = CblasNoTrans;
    transB = CblasNoTrans;
    gemm_m = seq_len_;
    gemm_n = head_dim_;
    gemm_k = seq_len_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA,
                     transB,
                     gemm_m,
                     gemm_n,
                     gemm_k,
                     alpha,
                     qk_out_grad_data,
                     k_ptr,
                     beta,
                     q_grad_ptr,
                     gemm_batch_size,
                     stride_a,
                     stride_b);

    // transpose bw
    std::vector<int> perm_1 = {1, 3, 0, 2, 4};
    phi::funcs::TransposeGPUKernelDriver<T>(
        dev_ctx_, *transpose_2_out_grad_tensor, perm_1, qkv_input_grad_tensor);
  }

 private:
  const phi::GPUContext& dev_ctx_;

  int64_t batch_size_;
  int64_t seq_len_;
  int64_t num_head_;
  int64_t head_dim_;

  AttnDropoutParam dropout_param_;
};

}  // namespace fusion
}  // namespace phi
