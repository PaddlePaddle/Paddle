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

#pragma once

#include "paddle/fluid/operators/dropout_impl.cu.h"
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/transpose_op.cu.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

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
  AttnDropoutParam(bool is_test, const std::string dropout_implementation,
                   float dropout_prob, bool is_upscale_in_train,
                   bool is_fix_seed, int seed_val, const Tensor* seed) {
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
  const Tensor* seed_;
};

template <typename T>
class FMHARef {
 public:
  FMHARef(const platform::CUDADeviceContext& dev_ctx, int64_t batch_size,
          int64_t seq_len, int64_t num_head, int64_t head_dim,
          AttnDropoutParam param)
      : dev_ctx_(dev_ctx),
        batch_size_(batch_size),
        seq_len_(seq_len),
        num_head_(num_head),
        head_dim_(head_dim),
        dropout_param_(param) {}

  ~FMHARef() {}

  void ComputeForward(const Tensor& qkv_input_tensor,
                      const Tensor* cache_kv_tensor,
                      const Tensor* src_mask_tensor,
                      Tensor* transpose_2_out_tensor,
                      Tensor* cache_kv_out_tensor, Tensor* qk_out_tensor,
                      Tensor* src_mask_out_tensor, Tensor* softmax_out_tensor,
                      Tensor* dropout_mask_out_tensor,
                      Tensor* dropout_out_tensor, Tensor* qktv_out_tensor,
                      Tensor* fmha_out_tensor) {
    // input shape: [bs, seq_len, 3, num_head, head_dim]
    // transpose with perm [2, 0, 3, 1, 4],
    // output_shape: [3, bs, num_head, seq_len, head_dim]
    int ndims = 5;
    std::vector<int> perm_1 = {2, 0, 3, 1, 4};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, qkv_input_tensor, perm_1,
                                transpose_2_out_tensor);
    T* qkv_data = transpose_2_out_tensor->data<T>();
    T* qk_out_data = qk_out_tensor->data<T>();
    T* qktv_out_data = qktv_out_tensor->data<T>();
    T* softmax_out_data = softmax_out_tensor->data<T>();
    T* dropout_out_data = dropout_out_tensor->data<T>();
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
      std::vector<const framework::Tensor*> ins = {&q_tensor};
      std::vector<framework::Tensor*> outs = {&q_tensor};
      paddle::operators::LaunchSameDimsElementwiseCudaKernel<T>(dev_ctx_, ins,
                                                                &outs, functor);
    }

    // q*k^t, batched_gemm
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasTrans;
    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    int gemm_batch_size = batch_size_ * num_head_;
    int gemm_m = seq_len_;
    int gemm_n = out_seq_len;
    int gemm_k = head_dim_;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    int64_t stride_a = gemm_m * gemm_k;
    int64_t stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha, q_ptr,
                     k_ptr, beta, qk_out_data, gemm_batch_size, stride_a,
                     stride_b);
    int softmax_axis = -1;
    if (src_mask_tensor != nullptr) {
      std::vector<const Tensor*> ins;
      std::vector<Tensor*> outs;
      ins.emplace_back(qk_out_tensor);
      ins.emplace_back(src_mask_tensor);
      outs.emplace_back(src_mask_out_tensor);
      int elewise_add_axis = -1;
      paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary,
                                                     T, T>(
          dev_ctx_, ins, &outs, elewise_add_axis, AddFunctor<T>());

      phi::SoftmaxForwardCUDAKernelDriver<T>(dev_ctx_, *src_mask_out_tensor,
                                             softmax_axis, softmax_out_tensor);
    } else {
      phi::SoftmaxForwardCUDAKernelDriver<T>(dev_ctx_, *qk_out_tensor,
                                             softmax_axis, softmax_out_tensor);
    }

    transB = CblasNoTrans;
    gemm_m = seq_len_;
    gemm_n = head_dim_;
    gemm_k = out_seq_len;
    alpha = static_cast<T>(1.0);
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;

    if (dropout_param_.dropout_prob_) {
      DropoutFwGPUKernelDriver<T>(
          static_cast<const phi::GPUContext&>(dev_ctx_),
          dropout_param_.is_test_, static_cast<const std::string>(
                                       dropout_param_.dropout_implementation_),
          dropout_param_.dropout_prob_, dropout_param_.is_upscale_in_train_,
          dropout_param_.is_fix_seed_, dropout_param_.seed_val_,
          static_cast<const Tensor&>(*softmax_out_tensor), dropout_param_.seed_,
          dropout_mask_out_tensor, dropout_out_tensor);
      blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                       dropout_out_data, v_ptr, beta, qktv_out_data,
                       gemm_batch_size, stride_a, stride_b);
    } else {
      // softmax_out * v, batched_gemm
      // output shape: [batch_size, num_heads, seq_len, head_dim]
      blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                       softmax_out_data, v_ptr, beta, qktv_out_data,
                       gemm_batch_size, stride_a, stride_b);
    }
    // transpose: [0, 2, 1, 3]
    // output shape: [batch_size, seq_len, num_heads, head_dim]
    std::vector<int> perm_3 = {0, 2, 1, 3};
    ndims = 4;
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, *qktv_out_tensor, perm_3,
                                fmha_out_tensor);
  }

  void ComputeBackward(
      const Tensor& transpose_2_out_tensor, const Tensor* src_mask_tensor,
      const Tensor& softmax_out_tensor, const Tensor& dropout_mask_out_tensor,
      const Tensor& dropout_out_tensor, const Tensor& qk_out_tensor,
      const Tensor& src_mask_out_tensor, const Tensor& fmha_out_grad_tensor,
      Tensor* qktv_out_grad_tensor, Tensor* dropout_out_grad_tensor,
      Tensor* softmax_out_grad_tensor, Tensor* src_mask_out_grad_tensor,
      Tensor* qk_out_grad_tensor, Tensor* transpose_2_out_grad_tensor,
      Tensor* src_mask_grad_tensor, Tensor* qkv_input_grad_tensor) {
    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
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
    const T* dropout_out_data = dropout_out_tensor.data<T>();
    T* dropout_out_grad_data = dropout_out_grad_tensor->data<T>();
    T* qktv_out_grad_data = qktv_out_grad_tensor->data<T>();

    // transpose bw
    int ndims = 4;
    std::vector<int> perm_3 = {0, 2, 1, 3};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, fmha_out_grad_tensor, perm_3,
                                qktv_out_grad_tensor);

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
      blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                       dropout_out_data, qktv_out_grad_data, beta, v_grad_ptr,
                       gemm_batch_size, stride_a, stride_b);
    } else {
      blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                       softmax_out_data, qktv_out_grad_data, beta, v_grad_ptr,
                       gemm_batch_size, stride_a, stride_b);
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
      blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                       qktv_out_grad_data, v_ptr, beta, dropout_out_grad_data,
                       gemm_batch_size, stride_a, stride_b);
    } else {
      blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                       qktv_out_grad_data, v_ptr, beta, softmax_out_grad_data,
                       gemm_batch_size, stride_a, stride_b);
    }
    // dropout bw
    if (dropout_param_.dropout_prob_) {
      DropoutGradGPUKernelDriver<T>(
          static_cast<const phi::GPUContext&>(dev_ctx_),
          static_cast<const std::string>(
              dropout_param_.dropout_implementation_),
          dropout_param_.dropout_prob_,
          static_cast<const Tensor&>(*dropout_out_grad_tensor),
          dropout_mask_out_tensor, softmax_out_grad_tensor->numel(),
          softmax_out_grad_tensor);
    }

    if (src_mask_tensor != nullptr) {
      phi::SoftmaxBackwardCUDAKernelDriver<T>(
          dev_ctx_, softmax_out_tensor, *softmax_out_grad_tensor, softmax_axis,
          src_mask_out_grad_tensor);

      // recall LaunchElementwiseCudaKernel fw:  src_mask_out = qk_out +
      // src_mask
      // Special case when dy is not needed and dx doesn't reduce
      if (qk_out_grad_tensor != nullptr && src_mask_grad_tensor == nullptr &&
          qk_out_tensor.dims() == src_mask_out_tensor.dims()) {
        VLOG(4) << "Special case when dy is not needed and dx doesn't "
                   "reduce";
        framework::TensorCopy(*src_mask_out_grad_tensor, dev_ctx_.GetPlace(),
                              dev_ctx_, qk_out_grad_tensor);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Only used for the backward elementwise_add op when"
            "dy is not needed and dx is not reduce"));
        return;
      }

    } else {
      phi::SoftmaxBackwardCUDAKernelDriver<T>(dev_ctx_, softmax_out_tensor,
                                              *softmax_out_grad_tensor,
                                              softmax_axis, qk_out_grad_tensor);
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
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qk_out_grad_data, q_ptr, beta, k_grad_ptr, gemm_batch_size,
                     stride_a, stride_b);
    // dx (seq_len * head_dim) = dout * y
    alpha = static_cast<T>(1.0 / sqrt(head_dim_));
    transA = CblasNoTrans;
    transB = CblasNoTrans;
    gemm_m = seq_len_;
    gemm_n = head_dim_;
    gemm_k = seq_len_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qk_out_grad_data, k_ptr, beta, q_grad_ptr, gemm_batch_size,
                     stride_a, stride_b);

    // transpose bw
    ndims = 5;
    std::vector<int> perm_1 = {1, 3, 0, 2, 4};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, *transpose_2_out_grad_tensor,
                                perm_1, qkv_input_grad_tensor);
  }

 private:
  const platform::CUDADeviceContext& dev_ctx_;

  int64_t batch_size_;
  int64_t seq_len_;
  int64_t num_head_;
  int64_t head_dim_;

  AttnDropoutParam dropout_param_;
};

template <typename T>
struct TernaryAddFunctor {
  inline HOSTDEVICE T operator()(T a, T b, T c) const { return a + b + c; }
};

template <typename T>
class FMHAGateRef {
 public:
  FMHAGateRef(const platform::CUDADeviceContext& dev_ctx, int64_t batch_size,
              int64_t seq_len_m, int64_t seq_len_r, int64_t num_head,
              int64_t head_dim)
      : dev_ctx_(dev_ctx),
        batch_size_(batch_size),
        seq_len_m_(seq_len_m),
        seq_len_r_(seq_len_r),
        num_head_(num_head),
        head_dim_(head_dim) {}

  void ComputeForward(const Tensor* nonbatched_bias, const Tensor& qkv_out,
                      const Tensor* src_mask, Tensor* qkv_transpose_out,
                      Tensor* softmax_out, Tensor* qktv_out, Tensor* fmha_out) {
    ComputeQKVTransposeForward(qkv_out, qkv_transpose_out);

    Tensor qk_out;
    qk_out.Resize({batch_size_, seq_len_m_, num_head_, seq_len_r_, seq_len_r_});
    T* qk_out_data = qk_out.mutable_data<T>(dev_ctx_.GetPlace());

    T* qkv_data = qkv_transpose_out->data<T>();
    T* qktv_out_data = qktv_out->data<T>();
    T* fmha_out_data = fmha_out->data<T>();

    int64_t q_size =
        batch_size_ * seq_len_m_ * seq_len_r_ * num_head_ * head_dim_;
    int64_t k_size = q_size;
    T* q_ptr = qkv_data;
    T* k_ptr = q_ptr + q_size;
    T* v_ptr = k_ptr + k_size;

    // q*k^t, batched_gemm
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasTrans;
    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    int gemm_batch_size = batch_size_ * seq_len_m_ * num_head_;
    int gemm_m = seq_len_r_;
    int gemm_n = seq_len_r_;
    int gemm_k = head_dim_;

    T alpha = static_cast<T>(1.0 / sqrt(head_dim_));
    T beta = static_cast<T>(0.0);
    int64_t stride_a = gemm_m * gemm_k;
    int64_t stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha, q_ptr,
                     k_ptr, beta, qk_out_data, gemm_batch_size, stride_a,
                     stride_b);

    ComputeBiasMaskSoftmaxForward(&qk_out, nonbatched_bias, src_mask,
                                  softmax_out);

    transB = CblasNoTrans;
    gemm_m = seq_len_r_;
    gemm_n = head_dim_;
    gemm_k = seq_len_r_;
    alpha = static_cast<T>(1.0);
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;

    T* softmax_out_data = softmax_out->data<T>();
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     softmax_out_data, v_ptr, beta, qktv_out_data,
                     gemm_batch_size, stride_a, stride_b);

    ComputeQKTVTransposeForward(*qktv_out, fmha_out);
  }

  void ComputeBackward(const Tensor& qkv_transpose_out, const Tensor* src_mask,
                       const Tensor& softmax_out, const Tensor& fmha_out_grad,
                       const Tensor* nonbatched_bias,
                       Tensor* nonbatched_bias_grad, Tensor* qktv_out_grad,
                       Tensor* softmax_out_grad, Tensor* src_mask_grad,
                       Tensor* qkv_transpose_out_grad, Tensor* qkv_out_grad) {
    ComputeQKTVTransposeBackward(fmha_out_grad, qktv_out_grad);

    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    int q_size = batch_size_ * seq_len_m_ * seq_len_r_ * num_head_ * head_dim_;
    int k_size = q_size;

    T* qkv_grad_ptr = qkv_transpose_out_grad->data<T>();
    T* q_grad_ptr = qkv_grad_ptr;
    T* k_grad_ptr = q_grad_ptr + q_size;
    T* v_grad_ptr = k_grad_ptr + k_size;

    const T* qkv_data = qkv_transpose_out.data<T>();
    const T* q_ptr = qkv_data;
    const T* k_ptr = q_ptr + q_size;
    const T* v_ptr = k_ptr + k_size;

    const T* softmax_out_data = softmax_out.data<T>();
    T* softmax_out_grad_data = softmax_out_grad->data<T>();
    T* qktv_out_grad_data = qktv_out_grad->data<T>();

    // softmax_out_data(x) * v_ptr(y) = qktv_out_grad_data
    CBLAS_TRANSPOSE transA = CblasTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    int gemm_batch_size = batch_size_ * seq_len_m_ * num_head_;
    int gemm_m = seq_len_r_;
    int gemm_n = head_dim_;
    int gemm_k = seq_len_r_;
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);
    int64_t stride_a = gemm_m * gemm_k;
    int64_t stride_b = gemm_k * gemm_n;

    // bw: dy = x^t * dout
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     softmax_out_data, qktv_out_grad_data, beta, v_grad_ptr,
                     gemm_batch_size, stride_a, stride_b);

    // // bw: dx = dout * y^t
    transA = CblasNoTrans;
    transB = CblasTrans;
    gemm_m = seq_len_r_;
    gemm_n = seq_len_r_;
    gemm_k = head_dim_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;

    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qktv_out_grad_data, v_ptr, beta, softmax_out_grad_data,
                     gemm_batch_size, stride_a, stride_b);

    Tensor qk_out_grad;
    qk_out_grad.Resize(
        {batch_size_, seq_len_m_, num_head_, seq_len_r_, seq_len_r_});
    T* qk_out_grad_data = qk_out_grad.mutable_data<T>(dev_ctx_.GetPlace());

    ComputeBiasMaskSoftmaxBackward(softmax_out_grad, &softmax_out, src_mask,
                                   src_mask_grad, &qk_out_grad,
                                   nonbatched_bias_grad);

    alpha = static_cast<T>(1.0);
    // recall batchedgemm(nt) fw:  q_ptr * (k_ptr)^t = qk_out
    // bw: dy (seq_len * head_dim) = (dout)^t * x
    transA = CblasTrans;
    transB = CblasNoTrans;
    gemm_m = seq_len_r_;
    gemm_n = head_dim_;
    gemm_k = seq_len_r_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qk_out_grad_data, q_ptr, beta, k_grad_ptr, gemm_batch_size,
                     stride_a, stride_b);

    // dx (seq_len * head_dim) = dout * y
    alpha = static_cast<T>(1.0 / sqrt(head_dim_));
    transA = CblasNoTrans;
    transB = CblasNoTrans;
    gemm_m = seq_len_r_;
    gemm_n = head_dim_;
    gemm_k = seq_len_r_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qk_out_grad_data, k_ptr, beta, q_grad_ptr, gemm_batch_size,
                     stride_a, stride_b);

    ComputeQKVTransposeBackward(*qkv_transpose_out_grad, qkv_out_grad);
  }

  // [batch_size, seq_len_m, seq_len_r, 3, num_head, c] ->
  //         [3, batch_size, seq_len_m, num_head, seq_len_r, c]
  void ComputeQKVTransposeForward(const Tensor& qkv_out,
                                  Tensor* qkv_transpose_out) {
    int ndims = 6;
    std::vector<int> perm = {3, 0, 1, 4, 2, 5};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, qkv_out, perm,
                                qkv_transpose_out);
  }

  void ComputeQKVTransposeBackward(const Tensor& qkv_transpose_out_grad,
                                   Tensor* qkv_out_grad) {
    int ndims = 6;
    std::vector<int> perm = {1, 2, 4, 0, 3, 5};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, qkv_transpose_out_grad, perm,
                                qkv_out_grad);
  }

  // [batch_size, seq_len_m, num_head, seq_len_r, c] ->
  //         [batch_size, seq_len_m, seq_len_r, num_head, c]
  void ComputeQKTVTransposeForward(const Tensor& qktv_out, Tensor* fmha_out) {
    int ndims = 5;
    std::vector<int> perm = {0, 1, 3, 2, 4};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, qktv_out, perm, fmha_out);
  }

  void ComputeQKTVTransposeBackward(const Tensor& fmha_out_grad,
                                    Tensor* qktv_out_grad) {
    int ndims = 5;
    std::vector<int> perm = {0, 1, 3, 2, 4};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, fmha_out_grad, perm,
                                qktv_out_grad);
  }

  // src_mask_out = qk_out + nonbatched_bias + src_mask
  // softmax_out = softmax(src_mask_out)
  void ComputeBiasMaskSoftmaxForward(const Tensor* qk_out,
                                     const Tensor* nonbatched_bias,
                                     const Tensor* src_mask,
                                     Tensor* softmax_out) {
    Tensor src_mask_out;
    src_mask_out.Resize(qk_out->dims());
    src_mask_out.mutable_data<T>(dev_ctx_.GetPlace());

    if (nonbatched_bias) {
      std::vector<const Tensor*> ins = {qk_out, nonbatched_bias, src_mask};
      std::vector<Tensor*> outs = {&src_mask_out};
      paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kTernary,
                                                     T, T>(
          dev_ctx_, ins, &outs, -1, TernaryAddFunctor<T>());

    } else {
      std::vector<const Tensor*> ins = {qk_out, src_mask};
      std::vector<Tensor*> outs = {&src_mask_out};
      paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary,
                                                     T, T>(dev_ctx_, ins, &outs,
                                                           -1, AddFunctor<T>());
    }

    phi::SoftmaxForwardCUDAKernelDriver<T>(dev_ctx_, src_mask_out, -1,
                                           softmax_out);
  }

  void ComputeBiasMaskSoftmaxBackward(const Tensor* softmax_out_grad,
                                      const Tensor* softmax_out,
                                      const Tensor* src_mask,
                                      Tensor* src_mask_grad,
                                      Tensor* qk_out_grad,
                                      Tensor* nonbatched_bias_grad) {
    Tensor* softmax_x_grad = nullptr;
    Tensor src_mask_out_grad;
    if (src_mask) {
      // src_mask_out = qk_out + src_mask
      // Special case when dy is not needed and dx doesn't reduce
      if (qk_out_grad && src_mask_grad == nullptr &&
          qk_out_grad->dims() == softmax_out->dims()) {
        VLOG(4) << "Special case when dy is not needed and dx doesn't "
                   "reduce";
        softmax_x_grad = qk_out_grad;
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Only used for the backward elementwise_add op when"
            "dy is not needed and dx is not reduce"));
        return;
      }
    } else {
      src_mask_out_grad.Resize(softmax_out->dims());
      src_mask_out_grad.mutable_data<T>(dev_ctx_.GetPlace());
      softmax_x_grad = &src_mask_out_grad;
    }

    phi::SoftmaxBackwardCUDAKernelDriver<T>(dev_ctx_, *softmax_out,
                                            *softmax_out_grad, -1, qk_out_grad);

    // [1, bs, num_head, seq_l, seq_l] -> [bs, num_head, seq_l, seq_l]
    if (nonbatched_bias_grad) {
      gpuStream_t stream = dev_ctx_.stream();
      TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dev_ctx_, *softmax_x_grad, nonbatched_bias_grad,
          kps::IdentityFunctor<T>(), {0, 1}, stream);
    }
  }

 private:
  const platform::CUDADeviceContext& dev_ctx_;

  int64_t batch_size_;
  int64_t seq_len_m_;
  int64_t seq_len_r_;
  int64_t num_head_;
  int64_t head_dim_;
};

}  // namespace operators
}  // namespace paddle
