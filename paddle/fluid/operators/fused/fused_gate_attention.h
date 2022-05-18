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

#include "paddle/fluid/operators/elementwise/elementwise_op_broadcast.cu.h"
#include "paddle/fluid/operators/transpose_op.cu.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct TernaryAddFunctor {
  inline HOSTDEVICE T operator()(T a, T b, T c) const { return a + b + c; }
};

template <typename T>
struct GateAttentionConfig {
  int64_t batch_size;
  int64_t seq_len_m;
  int64_t seq_len_r;
  int64_t query_dim;
  int64_t key_dim;
  int64_t kv_dim;
  int64_t m_size;
  int64_t num_heads;

  GateAttentionConfig(const Tensor* query, const Tensor* key,
                      const Tensor* query_weight, const Tensor* qkv_weight,
                      bool merge_qkv) {
    // query: shape=[batch_size, seq_len_m, seq_len_r, qkv_dim]
    batch_size = query->dims()[0];
    seq_len_m = query->dims()[1];
    seq_len_r = query->dims()[2];
    query_dim = query->dims()[3];

    if (merge_qkv) {
      PADDLE_ENFORCE_NOT_NULL(qkv_weight);

      // qkv_weight: shape=[3, num_heads, key_dim, qkv_dim]
      num_heads = qkv_weight->dims()[1];
      key_dim = qkv_weight->dims()[2];
      m_size = seq_len_r;
      kv_dim = query_dim;
    } else {
      PADDLE_ENFORCE_NOT_NULL(key);
      PADDLE_ENFORCE_NOT_NULL(query_weight);

      num_heads = query_weight->dims()[1];
      key_dim = query_weight->dims()[2];
      m_size = key->dims()[2];
      kv_dim = key->dims()[3];
    }
  }

  Tensor* GetQKVTransposeOut(const platform::CUDADeviceContext& dev_ctx) {
    qkv_transpose_out.Resize(
        {3, batch_size, seq_len_m, num_heads, seq_len_r, query_dim});
    VLOG(3) << "qkv_transpose_out.shape=[" << qkv_transpose_out.dims()
            << "], size=" << qkv_transpose_out.numel() * sizeof(T) / 1.0E6
            << " MB.";
    qkv_transpose_out.mutable_data<T>(dev_ctx.GetPlace());
    return &qkv_transpose_out;
  }

  Tensor* GetQTransposeOut(const platform::CUDADeviceContext& dev_ctx) {
    q_transpose_out.Resize(
        {batch_size, seq_len_m, num_heads, seq_len_r, key_dim});
    VLOG(3) << "q_transpose_out.shape=[" << q_transpose_out.dims()
            << "], size=" << q_transpose_out.numel() * sizeof(T) / 1.0E6
            << " MB.";
    q_transpose_out.mutable_data<T>(dev_ctx.GetPlace());
    return &q_transpose_out;
  }

  Tensor* GetKTransposeOut(const platform::CUDADeviceContext& dev_ctx) {
    k_transpose_out.Resize({batch_size, seq_len_m, num_heads, m_size, key_dim});
    VLOG(3) << "k_transpose_out.shape=[" << k_transpose_out.dims()
            << "], size=" << k_transpose_out.numel() * sizeof(T) / 1.0E6
            << " MB.";
    k_transpose_out.mutable_data<T>(dev_ctx.GetPlace());
    return &k_transpose_out;
  }

  Tensor* GetVTransposeOut(const platform::CUDADeviceContext& dev_ctx) {
    v_transpose_out.Resize({batch_size, seq_len_m, num_heads, m_size, key_dim});
    VLOG(3) << "v_transpose_out.shape=[" << v_transpose_out.dims()
            << "], size=" << v_transpose_out.numel() * sizeof(T) / 1.0E6
            << " MB.";
    v_transpose_out.mutable_data<T>(dev_ctx.GetPlace());
    return &v_transpose_out;
  }

  Tensor* GetQKOut(const platform::CUDADeviceContext& dev_ctx,
                   Tensor* softmax_out) {
    // softmax_dim = qk_out_dim[-1] = qk_out_dim[rank - 1]
    int softmax_dim = m_size;
    if (phi::UseCudnnSoftmax<T>(dev_ctx, softmax_dim, true)) {
      // Not sure whether cudnn softmax can execute inplace.
      qk_out.Resize({batch_size, seq_len_m, num_heads, seq_len_r, m_size});
      VLOG(3) << "qk_out.shape=[" << qk_out.dims()
              << "], size=" << qk_out.numel() * sizeof(T) / 1.0E6 << " MB.";
      qk_out.mutable_data<T>(dev_ctx.GetPlace());
      return &qk_out;
    } else {
      return softmax_out;
    }
  }

  Tensor* GetFMHAOut(const platform::CUDADeviceContext& dev_ctx) {
    fmha_out.Resize({batch_size, seq_len_m, seq_len_r, num_heads, key_dim});
    VLOG(3) << "[FusedGateAttentionOpKernel] fmha_out.shape=["
            << fmha_out.dims()
            << "], size=" << fmha_out.numel() * sizeof(T) / 1.0E6 << " MB.";
    fmha_out.mutable_data<T>(dev_ctx.GetPlace());
    return &fmha_out;
  }

  void ClearQKVTransposeOut() {
    if (qkv_transpose_out.IsInitialized()) {
      qkv_transpose_out.clear();
    }
  }

  void ClearQKOut() {
    if (qk_out.IsInitialized()) {
      qk_out.clear();
    }
  }

 private:
  // qkv_transpose_out: shape=[3, batch_size, seq_len_m, num_heads, seq_len_r,
  // q_dim]
  Tensor qkv_transpose_out;
  // QKV is not merged
  Tensor q_transpose_out;
  Tensor k_transpose_out;
  Tensor v_transpose_out;
  // qk_out = BatchedGEMM(Q, K^T)
  // qk_out: shape=[batch_size, seq_len_m, num_heads, seq_len_r, m_size]
  // softmax_out = softmax(qk_out + nonbatched_bias + src_mask)
  // The shape of qk_out, softmax_out is the same, thus can be called inplace.
  Tensor qk_out;
  // fmha_out = transpose(qktv_out)
  Tensor fmha_out;
};

template <typename T>
class FMHAGateRef {
 public:
  FMHAGateRef(const platform::CUDADeviceContext& dev_ctx, bool merge_qkv,
              GateAttentionConfig<T>* config)
      : dev_ctx_(dev_ctx), merge_qkv_(merge_qkv), config_(config) {
    batch_size_ = config_->batch_size;
    seq_len_m_ = config_->seq_len_m;
    seq_len_r_ = config_->seq_len_r;
    q_dim_ = config_->query_dim;
    key_dim_ = config_->key_dim;
    kv_dim_ = config_->kv_dim;
    m_size_ = config_->m_size;
    num_head_ = config_->num_heads;
  }

  void ComputeForward(const Tensor* query_out, const Tensor* key_out,
                      const Tensor* value_out, const Tensor* qkv_out,
                      const Tensor* nonbatched_bias, const Tensor* src_mask,
                      Tensor* softmax_out, Tensor* qktv_out, Tensor* fmha_out) {
    T* q_ptr = nullptr;
    T* k_ptr = nullptr;
    T* v_ptr = nullptr;
    if (merge_qkv_) {
      // qkv_transpose_out = transpose(qkv_out)
      PADDLE_ENFORCE_NOT_NULL(qkv_out);
      Tensor* qkv_transpose_out = config_->GetQKVTransposeOut(dev_ctx_);
      ComputeQKVTransposeForward(*qkv_out, qkv_transpose_out);

      // q_size == k_size
      int64_t q_size =
          batch_size_ * seq_len_m_ * seq_len_r_ * num_head_ * key_dim_;
      q_ptr = qkv_transpose_out->data<T>();
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + q_size;
    } else {
      PADDLE_ENFORCE_NOT_NULL(query_out);
      PADDLE_ENFORCE_NOT_NULL(key_out);
      PADDLE_ENFORCE_NOT_NULL(value_out);

      Tensor* q_transpose_out = config_->GetQTransposeOut(dev_ctx_);
      Tensor* k_transpose_out = config_->GetKTransposeOut(dev_ctx_);
      Tensor* v_transpose_out = config_->GetVTransposeOut(dev_ctx_);
      ComputeQKVTransposeForward(*query_out, *key_out, *value_out,
                                 q_transpose_out, k_transpose_out,
                                 v_transpose_out);

      // q_size != k_size
      q_ptr = q_transpose_out->data<T>();
      k_ptr = k_transpose_out->data<T>();
      v_ptr = v_transpose_out->data<T>();
    }

    // qk_out = BatchedGEMM(Q, K^T)
    // [bs, s_m, s_r, num_head, key_dim] * [bs, s_m, num_head, m_size, key_dim]
    Tensor* qk_out = config_->GetQKOut(dev_ctx_, softmax_out);
    T* qk_out_ptr = qk_out->data<T>();

    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    int gemm_batch_size = batch_size_ * seq_len_m_ * num_head_;
    int gemm_m = seq_len_r_;
    int gemm_n = m_size_;
    int gemm_k = key_dim_;

    T alpha = static_cast<T>(1.0 / sqrt(key_dim_));
    T beta = static_cast<T>(0.0);
    int64_t stride_a = gemm_m * gemm_k;
    int64_t stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(CblasNoTrans, CblasTrans, gemm_m, gemm_n, gemm_k, alpha,
                     q_ptr, k_ptr, beta, qk_out_ptr, gemm_batch_size, stride_a,
                     stride_b);

    ComputeBiasMaskSoftmaxForward(nonbatched_bias, src_mask, qk_out,
                                  softmax_out);
    config_->ClearQKOut();

    // qk * v
    // [bs, s_m, num_head, s_r, m_size] * [bs, s_m, num_head, m_size, key_dim]
    gemm_m = seq_len_r_;
    gemm_n = key_dim_;
    gemm_k = m_size_;
    alpha = static_cast<T>(1.0);
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;

    T* softmax_out_data = softmax_out->data<T>();
    T* qktv_out_data = qktv_out->data<T>();
    blas.BatchedGEMM(CblasNoTrans, CblasNoTrans, gemm_m, gemm_n, gemm_k, alpha,
                     softmax_out_data, v_ptr, beta, qktv_out_data,
                     gemm_batch_size, stride_a, stride_b);
    config_->ClearQKVTransposeOut();

    ComputeQKTVTransposeForward(*qktv_out, fmha_out);
  }

  void ComputeBackward(
      const Tensor& q_transpose_out, const Tensor& k_transpose_out,
      const Tensor& v_transpose_out, const Tensor& qkv_transpose_out,
      const Tensor& softmax_out, const Tensor& fmha_out_grad,
      const Tensor* nonbatched_bias, Tensor* nonbatched_bias_grad,
      Tensor* qktv_out_grad, Tensor* softmax_out_grad, Tensor* src_mask_grad,
      Tensor* q_transpose_out_grad, Tensor* k_transpose_out_grad,
      Tensor* v_transpose_out_grad, Tensor* q_out_grad, Tensor* k_out_grad,
      Tensor* v_out_grad, Tensor* qkv_transpose_out_grad,
      Tensor* qkv_out_grad) {
    ComputeQKTVTransposeBackward(fmha_out_grad, qktv_out_grad);

    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    const T* q_ptr = nullptr;
    const T* k_ptr = nullptr;
    const T* v_ptr = nullptr;
    T* q_grad_ptr = nullptr;
    T* k_grad_ptr = nullptr;
    T* v_grad_ptr = nullptr;

    if (merge_qkv_) {
      int q_size = batch_size_ * seq_len_m_ * seq_len_r_ * num_head_ * key_dim_;
      int k_size = q_size;

      T* qkv_grad_ptr = qkv_transpose_out_grad->data<T>();
      q_grad_ptr = qkv_grad_ptr;
      k_grad_ptr = q_grad_ptr + q_size;
      v_grad_ptr = k_grad_ptr + k_size;

      const T* qkv_data = qkv_transpose_out.data<T>();
      q_ptr = qkv_data;
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + k_size;
    } else {
      q_grad_ptr = q_transpose_out_grad->data<T>();
      k_grad_ptr = k_transpose_out_grad->data<T>();
      v_grad_ptr = v_transpose_out_grad->data<T>();

      q_ptr = q_transpose_out.data<T>();
      k_ptr = k_transpose_out.data<T>();
      v_ptr = v_transpose_out.data<T>();
    }

    const T* softmax_out_data = softmax_out.data<T>();
    T* softmax_out_grad_data = softmax_out_grad->data<T>();
    T* qktv_out_grad_data = qktv_out_grad->data<T>();

    // softmax_out_data(x) * v_ptr(y) = qktv_out_grad_data
    CBLAS_TRANSPOSE transA = CblasTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    int gemm_batch_size = batch_size_ * seq_len_m_ * num_head_;
    int gemm_m = m_size_;
    int gemm_n = key_dim_;
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
    gemm_n = m_size_;
    gemm_k = key_dim_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;

    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qktv_out_grad_data, v_ptr, beta, softmax_out_grad_data,
                     gemm_batch_size, stride_a, stride_b);

    Tensor qk_out_grad;
    qk_out_grad.Resize(
        {batch_size_, seq_len_m_, num_head_, seq_len_r_, m_size_});
    T* qk_out_grad_data = qk_out_grad.mutable_data<T>(dev_ctx_.GetPlace());

    ComputeBiasMaskSoftmaxBackward(softmax_out_grad, &softmax_out,
                                   src_mask_grad, &qk_out_grad,
                                   nonbatched_bias_grad);

    alpha = static_cast<T>(1.0 / sqrt(key_dim_));
    // recall batchedgemm(nt) fw:  q_ptr * (k_ptr)^t = qk_out
    // bw: dy (seq_len * head_dim) = (dout)^t * x
    transA = CblasTrans;
    transB = CblasNoTrans;
    gemm_m = m_size_;
    gemm_n = key_dim_;
    gemm_k = seq_len_r_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qk_out_grad_data, q_ptr, beta, k_grad_ptr, gemm_batch_size,
                     stride_a, stride_b);

    // dx (seq_len * head_dim) = dout * y
    transA = CblasNoTrans;
    transB = CblasNoTrans;
    gemm_m = seq_len_r_;
    gemm_n = key_dim_;
    gemm_k = m_size_;
    stride_a = gemm_m * gemm_k;
    stride_b = gemm_k * gemm_n;
    blas.BatchedGEMM(transA, transB, gemm_m, gemm_n, gemm_k, alpha,
                     qk_out_grad_data, k_ptr, beta, q_grad_ptr, gemm_batch_size,
                     stride_a, stride_b);
    if (merge_qkv_) {
      ComputeQKVTransposeBackward(*qkv_transpose_out_grad, qkv_out_grad);
    } else {
      ComputeQKVTransposeBackward(*q_transpose_out_grad, *k_transpose_out_grad,
                                  *v_transpose_out_grad, q_out_grad, k_out_grad,
                                  v_out_grad);
    }
  }

  void ComputeQKVTransposeForward(const Tensor& q_out, const Tensor& k_out,
                                  const Tensor& v_out, Tensor* q_transpose_out,
                                  Tensor* k_transpose_out,
                                  Tensor* v_transpose_out) {
    int ndims = 5;
    std::vector<int> perm = {0, 1, 3, 2, 4};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, q_out, perm, q_transpose_out);
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, k_out, perm, k_transpose_out);
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, v_out, perm, v_transpose_out);
  }

  void ComputeQKVTransposeBackward(const Tensor& q_transpose_out_grad,
                                   const Tensor& k_transpose_out_grad,
                                   const Tensor& v_transpose_out_grad,
                                   Tensor* q_out_grad, Tensor* k_out_grad,
                                   Tensor* v_out_grad) {
    int ndims = 5;
    std::vector<int> perm = {0, 1, 3, 2, 4};
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, q_transpose_out_grad, perm,
                                q_out_grad);
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, k_transpose_out_grad, perm,
                                k_out_grad);
    TransposeGPUKernelDriver<T>(dev_ctx_, ndims, v_transpose_out_grad, perm,
                                v_out_grad);
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

  // qk_out = qk_out + nonbatched_bias + src_mask
  // softmax_out = softmax(src_mask_out)
  void ComputeBiasMaskSoftmaxForward(const Tensor* nonbatched_bias,
                                     const Tensor* src_mask, Tensor* qk_out,
                                     Tensor* softmax_out) {
    if (nonbatched_bias) {
      std::vector<const Tensor*> ins = {qk_out, nonbatched_bias, src_mask};
      std::vector<Tensor*> outs = {qk_out};
      paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kTernary,
                                                     T, T>(
          dev_ctx_, ins, &outs, -1, TernaryAddFunctor<T>());
    } else {
      std::vector<const Tensor*> ins = {qk_out, src_mask};
      std::vector<Tensor*> outs = {qk_out};
      paddle::operators::LaunchElementwiseCudaKernel<ElementwiseType::kBinary,
                                                     T, T>(
          dev_ctx_, ins, &outs, -1, phi::funcs::AddFunctor<T>());
    }
    phi::SoftmaxForwardCUDAKernelDriver<T>(dev_ctx_, *qk_out, -1, softmax_out);
  }

  // src_mask_out = qk_out + nonbatched_bias + src_mask
  // softmax_out = softmax(src_mask_out)
  void ComputeBiasMaskSoftmaxBackward(const Tensor* softmax_out_grad,
                                      const Tensor* softmax_out,
                                      Tensor* src_mask_grad,
                                      Tensor* qk_out_grad,
                                      Tensor* nonbatched_bias_grad) {
    PADDLE_ENFORCE_NOT_NULL(qk_out_grad);

    PADDLE_ENFORCE_EQ(qk_out_grad->dims(), softmax_out->dims(),
                      platform::errors::InvalidArgument(
                          "The shape of qk_out_grad and softmax_out is "
                          "expected to be the same. But recieved qk_out_grad's "
                          "shape = %s, softmax_out's shape = %s.",
                          qk_out_grad->dims(), softmax_out->dims()));

    PADDLE_ENFORCE_EQ(src_mask_grad, nullptr,
                      platform::errors::InvalidArgument(
                          "src_mask_grad is expected to be nullptr."));

    phi::SoftmaxBackwardCUDAKernelDriver<T>(dev_ctx_, *softmax_out,
                                            *softmax_out_grad, -1, qk_out_grad);

    // [1, bs, num_head, seq_l, seq_l] -> [bs, num_head, seq_l, seq_l]
    if (nonbatched_bias_grad) {
      gpuStream_t stream = dev_ctx_.stream();
      TensorReduceImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dev_ctx_, *qk_out_grad, nonbatched_bias_grad,
          kps::IdentityFunctor<T>(), {0, 1}, stream);
    }
  }

 private:
  const platform::CUDADeviceContext& dev_ctx_;
  bool merge_qkv_;
  int64_t batch_size_;
  int64_t seq_len_m_;
  int64_t seq_len_r_;
  int64_t q_dim_;
  int64_t key_dim_;
  int64_t kv_dim_;
  int64_t m_size_;
  int64_t num_head_;
  GateAttentionConfig<T>* config_{nullptr};  // Not owned
};

}  // namespace operators
}  // namespace paddle
