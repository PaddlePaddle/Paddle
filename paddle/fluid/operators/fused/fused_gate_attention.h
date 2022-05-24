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
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/gpudnn/softmax_gpudnn.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

inline std::string MemoryDebugString(const Tensor& t) {
  std::stringstream ss;
  ss << "shape=[" << t.dims()
     << "], size=" << static_cast<float>(t.memory_size()) / (1 << 20)
     << " MB, ptr=" << t.data();

  size_t total = 0;
  size_t available = 0;
  platform::GpuMemoryUsage(&available, &total);
  ss << "; memory allocated="
     << static_cast<float>(total - available) / (1 << 20) << " MB";
  return ss.str();
}

template <typename T>
struct TernaryAddFunctor {
  inline HOSTDEVICE T operator()(T a, T b, T c) const { return a + b + c; }
};

template <typename T>
struct GateAttentionConfig {
 public:
  int64_t batch_size;
  int64_t seq_len_m;
  int64_t seq_len_r;
  int64_t q_dim;
  int64_t kv_dim;
  int64_t key_dim;
  int64_t m_size;
  int64_t num_heads;

  phi::DDim qkv_out_dims;
  phi::DDim qkv_transpose_out_dims;

  phi::DDim q_out_dims;
  phi::DDim kv_out_dims;
  phi::DDim q_transpose_out_dims;
  phi::DDim kv_transpose_out_dims;

  phi::DDim qk_out_dims;
  phi::DDim softmax_out_dims;
  phi::DDim qktv_out_dims;
  phi::DDim fmha_out_dims;
  phi::DDim gate_out_dims;

  GateAttentionConfig(const Tensor* query, const Tensor* key,
                      const Tensor* query_weight, const Tensor* qkv_weight,
                      bool merge_qkv) {
    // query: shape=[batch_size, seq_len_m, seq_len_r, q_dim]
    batch_size = query->dims()[0];
    seq_len_m = query->dims()[1];
    seq_len_r = query->dims()[2];
    q_dim = query->dims()[3];

    if (merge_qkv) {
      PADDLE_ENFORCE_NOT_NULL(qkv_weight);

      // When q_dim == kv_dim, QKV matmul can be computed merged.
      // qkv_weight: shape=[3, num_heads, key_dim, q_dim]
      num_heads = qkv_weight->dims()[1];
      key_dim = qkv_weight->dims()[2];
      m_size = seq_len_r;
      kv_dim = q_dim;

      qkv_out_dims = {batch_size, seq_len_m, seq_len_r, 3, num_heads, key_dim};
      qkv_transpose_out_dims = {3,         batch_size, seq_len_m,
                                num_heads, seq_len_r,  key_dim};
    } else {
      PADDLE_ENFORCE_NOT_NULL(key);
      PADDLE_ENFORCE_NOT_NULL(query_weight);

      // When q_dim != kv_dim, QKV matmul must be computed saparately.
      // key: shape=[batch_size, seq_len_m, m_size, kv_dim]
      // query_w: shape=[q_dim, num_heads, key_dim]
      num_heads = query_weight->dims()[1];
      key_dim = query_weight->dims()[2];
      m_size = key->dims()[2];
      kv_dim = key->dims()[3];

      q_out_dims = {batch_size, seq_len_m, seq_len_r, num_heads, key_dim};
      kv_out_dims = {batch_size, seq_len_m, m_size, num_heads, key_dim};
      q_transpose_out_dims = {batch_size, seq_len_m, num_heads, seq_len_r,
                              key_dim};
      kv_transpose_out_dims = {batch_size, seq_len_m, num_heads, m_size,
                               key_dim};
    }

    qk_out_dims = {batch_size, seq_len_m, num_heads, seq_len_r, m_size};
    softmax_out_dims = {batch_size, seq_len_m, num_heads, seq_len_r, m_size};
    qktv_out_dims = {batch_size, seq_len_m, num_heads, seq_len_r, key_dim};
    fmha_out_dims = {batch_size, seq_len_m, seq_len_r, num_heads, key_dim};
    gate_out_dims = {batch_size, seq_len_m, seq_len_r, num_heads, key_dim};
  }

  int64_t GetQuerySize() const {
    return batch_size * seq_len_m * seq_len_r * num_heads * key_dim;
  }

  Tensor* GetQKVTransposeOut(const platform::CUDADeviceContext& dev_ctx) {
    qkv_transpose_out.Resize(qkv_transpose_out_dims);
    qkv_transpose_out.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "qkv_transpose_out: " << MemoryDebugString(qkv_transpose_out);
    return &qkv_transpose_out;
  }

  Tensor* GetQTransposeOut(const platform::CUDADeviceContext& dev_ctx) {
    q_transpose_out.Resize(q_transpose_out_dims);
    q_transpose_out.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "q_transpose_out: " << MemoryDebugString(q_transpose_out);
    return &q_transpose_out;
  }

  Tensor* GetKTransposeOut(const platform::CUDADeviceContext& dev_ctx) {
    k_transpose_out.Resize(kv_transpose_out_dims);
    k_transpose_out.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "k_transpose_out: " << MemoryDebugString(k_transpose_out);
    return &k_transpose_out;
  }

  Tensor* GetVTransposeOut(const platform::CUDADeviceContext& dev_ctx) {
    v_transpose_out.Resize(kv_transpose_out_dims);
    v_transpose_out.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "v_transpose_out: " << MemoryDebugString(v_transpose_out);
    return &v_transpose_out;
  }

  Tensor* GetQKOut(const platform::CUDADeviceContext& dev_ctx,
                   Tensor* softmax_out) {
    // softmax_dim = qk_out_dim[-1] = qk_out_dim[rank - 1]
    int softmax_dim = m_size;
    if (!softmax_out || phi::UseCudnnSoftmax<T>(dev_ctx, softmax_dim, true)) {
      // Not sure whether cudnn softmax can execute inplace.
      qk_out.Resize(qk_out_dims);
      qk_out.mutable_data<T>(dev_ctx.GetPlace());
      VLOG(4) << "qk_out: " << MemoryDebugString(qk_out);
      return &qk_out;
    } else {
      return softmax_out;
    }
  }

  Tensor* GetFMHAOut(const platform::CUDADeviceContext& dev_ctx) {
    fmha_out.Resize(fmha_out_dims);
    fmha_out.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "fmha_out: " << MemoryDebugString(fmha_out);
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

 protected:
  // qkv_transpose_out: shape=
  //          [3, batch_size, seq_len_m, num_heads, seq_len_r, q_dim]
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
struct GateAttentionGradConfig : public GateAttentionConfig<T> {
 public:
  GateAttentionGradConfig(const Tensor* query, const Tensor* key,
                          const Tensor* query_weight, const Tensor* qkv_weight,
                          bool merge_qkv)
      : GateAttentionConfig<T>(query, key, query_weight, qkv_weight,
                               merge_qkv) {}

  Tensor* GetQKVTransposeOutGrad(const platform::CUDADeviceContext& dev_ctx) {
    qkv_transpose_out_grad.Resize(this->qkv_transpose_out_dims);
    qkv_transpose_out_grad.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "qkv_transpose_out_grad: "
            << MemoryDebugString(qkv_transpose_out_grad);
    return &qkv_transpose_out_grad;
  }

  Tensor* GetQTransposeOutGrad(const platform::CUDADeviceContext& dev_ctx) {
    q_transpose_out_grad.Resize(this->q_transpose_out_dims);
    q_transpose_out_grad.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "q_transpose_out_grad: "
            << MemoryDebugString(q_transpose_out_grad);
    return &q_transpose_out_grad;
  }

  Tensor* GetKTransposeOutGrad(const platform::CUDADeviceContext& dev_ctx) {
    k_transpose_out_grad.Resize(this->kv_transpose_out_dims);
    k_transpose_out_grad.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "k_transpose_out_grad: "
            << MemoryDebugString(k_transpose_out_grad);
    return &k_transpose_out_grad;
  }

  Tensor* GetVTransposeOutGrad(const platform::CUDADeviceContext& dev_ctx) {
    v_transpose_out_grad.Resize(this->kv_transpose_out_dims);
    v_transpose_out_grad.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "v_transpose_out_grad: "
            << MemoryDebugString(v_transpose_out_grad);
    return &v_transpose_out_grad;
  }

  Tensor* GetQKOutGrad(const platform::CUDADeviceContext& dev_ctx) {
    qk_out_grad.Resize(this->qk_out_dims);
    qk_out_grad.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "qk_out_grad: " << MemoryDebugString(qk_out_grad);
    return &qk_out_grad;
  }

  Tensor* GetFMHAOutGrad(const platform::CUDADeviceContext& dev_ctx) {
    fmha_out_grad.Resize(this->fmha_out_dims);
    fmha_out_grad.mutable_data<T>(dev_ctx.GetPlace());
    VLOG(4) << "fmha_out_grad: " << MemoryDebugString(fmha_out_grad);
    return &fmha_out_grad;
  }

 protected:
  Tensor qkv_transpose_out_grad;
  Tensor q_transpose_out_grad;
  Tensor k_transpose_out_grad;
  Tensor v_transpose_out_grad;
  Tensor qk_out_grad;
  Tensor fmha_out_grad;
};

template <typename T>
class FMHAGateRef {
 public:
  FMHAGateRef(const platform::CUDADeviceContext& dev_ctx, bool merge_qkv)
      : dev_ctx_(dev_ctx), merge_qkv_(merge_qkv) {}

  void ComputeForward(const Tensor* query_out, const Tensor* key_out,
                      const Tensor* value_out, const Tensor* qkv_out,
                      const Tensor* nonbatched_bias, const Tensor* src_mask,
                      Tensor* softmax_out, Tensor* qktv_out, Tensor* fmha_out,
                      GateAttentionConfig<T>* config) {
    T* q_ptr = nullptr;
    T* k_ptr = nullptr;
    T* v_ptr = nullptr;
    if (merge_qkv_) {
      // qkv_transpose_out = transpose(qkv_out)
      PADDLE_ENFORCE_NOT_NULL(qkv_out);
      Tensor* qkv_transpose_out = config->GetQKVTransposeOut(dev_ctx_);
      ComputeQKVTransposeForward(*qkv_out, qkv_transpose_out);

      // q_size == k_size
      int64_t q_size = config->GetQuerySize();
      q_ptr = qkv_transpose_out->data<T>();
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + q_size;
    } else {
      PADDLE_ENFORCE_NOT_NULL(query_out);
      PADDLE_ENFORCE_NOT_NULL(key_out);
      PADDLE_ENFORCE_NOT_NULL(value_out);

      Tensor* q_transpose_out = config->GetQTransposeOut(dev_ctx_);
      Tensor* k_transpose_out = config->GetKTransposeOut(dev_ctx_);
      Tensor* v_transpose_out = config->GetVTransposeOut(dev_ctx_);
      ComputeQKVTransposeForward(*query_out, *key_out, *value_out,
                                 q_transpose_out, k_transpose_out,
                                 v_transpose_out);

      // q_size != k_size
      q_ptr = q_transpose_out->data<T>();
      k_ptr = k_transpose_out->data<T>();
      v_ptr = v_transpose_out->data<T>();
    }

    // qk_out = BatchedGEMM(Q, K^T)
    // [batch_size, seq_len_m, num_heads, seq_len_r, key_dim] *
    //                [batch_size, seq_len_m, num_heads, m_size, key_dim]
    // -> [batch_size, seq_len_m, num_heads, seq_len_r, m_size]
    Tensor* qk_out = config->GetQKOut(dev_ctx_, softmax_out);
    T* qk_out_ptr = qk_out->data<T>();

    int64_t gemm_batch_size =
        config->batch_size * config->seq_len_m * config->num_heads;
    int64_t gemm_m = config->seq_len_r;
    int64_t gemm_n = config->m_size;
    int64_t gemm_k = config->key_dim;

    T alpha = static_cast<T>(1.0 / sqrt(config->key_dim));
    ComputeBatchedGEMM(q_ptr, k_ptr, qk_out_ptr, false, true, gemm_m, gemm_n,
                       gemm_k, gemm_batch_size, alpha);

    // softmax_out = softmax(qk_out + nonbatched_bias + src_mask)
    ComputeBiasMaskSoftmaxForward(nonbatched_bias, src_mask, qk_out,
                                  softmax_out);
    config->ClearQKOut();

    // qktv_out = BatchedGEMM(softmax_out, V)
    // [batch_size, seq_len_m, num_heads, seq_len_r, m_size] *
    //               [batch_size, seq_len_m, num_heads, m_size, key_dim]
    // -> [batch_size, seq_len_m, num_heads, seq_len_r, key_dim]
    gemm_m = config->seq_len_r;
    gemm_n = config->key_dim;
    gemm_k = config->m_size;

    T* softmax_out_ptr = softmax_out->data<T>();
    T* qktv_out_ptr = qktv_out->data<T>();
    ComputeBatchedGEMM(softmax_out_ptr, v_ptr, qktv_out_ptr, false, false,
                       gemm_m, gemm_n, gemm_k, gemm_batch_size);
    config->ClearQKVTransposeOut();

    ComputeQKTVTransposeForward(*qktv_out, fmha_out);
  }

  void ComputeBackward(const Tensor* query_out, const Tensor* key_out,
                       const Tensor* value_out, const Tensor* qkv_out,
                       const Tensor* softmax_out, const Tensor* fmha_out_grad,
                       Tensor* qktv_out_grad, Tensor* softmax_out_grad,
                       Tensor* src_mask_grad, Tensor* nonbatched_bias_grad,
                       Tensor* q_out_grad, Tensor* k_out_grad,
                       Tensor* v_out_grad, Tensor* qkv_out_grad,
                       GateAttentionGradConfig<T>* config) {
    // Forward: fmha_out = transpose(qktv_out)
    ComputeQKTVTransposeBackward(*fmha_out_grad, qktv_out_grad);

    const T* q_ptr = nullptr;
    const T* k_ptr = nullptr;
    const T* v_ptr = nullptr;

    T* q_grad_ptr = nullptr;
    T* k_grad_ptr = nullptr;
    T* v_grad_ptr = nullptr;

    Tensor* qkv_transpose_out_grad = nullptr;
    Tensor* q_transpose_out_grad = nullptr;
    Tensor* k_transpose_out_grad = nullptr;
    Tensor* v_transpose_out_grad = nullptr;
    if (merge_qkv_) {
      PADDLE_ENFORCE_NOT_NULL(qkv_out);

      // Re-compute qkv_transpose_out = transpose(qkv_out)
      Tensor* qkv_transpose_out = config->GetQKVTransposeOut(dev_ctx_);
      ComputeQKVTransposeForward(*qkv_out, qkv_transpose_out);

      int64_t q_size = config->GetQuerySize();
      q_ptr = qkv_transpose_out->data<T>();
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + q_size;

      qkv_transpose_out_grad = config->GetQKVTransposeOutGrad(dev_ctx_);
      q_grad_ptr = qkv_transpose_out_grad->data<T>();
      k_grad_ptr = q_grad_ptr + q_size;
      v_grad_ptr = k_grad_ptr + q_size;
    } else {
      PADDLE_ENFORCE_NOT_NULL(query_out);
      PADDLE_ENFORCE_NOT_NULL(key_out);
      PADDLE_ENFORCE_NOT_NULL(value_out);

      // Re-compute q_transpose_out, k_transpose_out, v_transpose_out
      Tensor* q_transpose_out = config->GetQTransposeOut(dev_ctx_);
      Tensor* k_transpose_out = config->GetKTransposeOut(dev_ctx_);
      Tensor* v_transpose_out = config->GetVTransposeOut(dev_ctx_);
      ComputeQKVTransposeForward(*query_out, *key_out, *value_out,
                                 q_transpose_out, k_transpose_out,
                                 v_transpose_out);

      q_ptr = q_transpose_out->data<T>();
      k_ptr = k_transpose_out->data<T>();
      v_ptr = v_transpose_out->data<T>();

      q_transpose_out_grad = config->GetQTransposeOutGrad(dev_ctx_);
      k_transpose_out_grad = config->GetKTransposeOutGrad(dev_ctx_);
      v_transpose_out_grad = config->GetVTransposeOutGrad(dev_ctx_);

      q_grad_ptr = q_transpose_out_grad->data<T>();
      k_grad_ptr = k_transpose_out_grad->data<T>();
      v_grad_ptr = v_transpose_out_grad->data<T>();
    }

    // Forward: qktv_out = BatchedGEMM(softmax_out, V)
    // Backward:
    //  V_grad = BatchedGEMM(softmax_out^T, qktv_out_grad) (dy = x^T * dout)
    int64_t gemm_batch_size =
        config->batch_size * config->seq_len_m * config->num_heads;
    int64_t gemm_m = config->m_size;
    int64_t gemm_n = config->key_dim;
    int64_t gemm_k = config->seq_len_r;

    const T* softmax_out_ptr = softmax_out->data<T>();
    T* qktv_out_grad_ptr = qktv_out_grad->data<T>();
    ComputeBatchedGEMM(softmax_out_ptr, qktv_out_grad_ptr, v_grad_ptr, true,
                       false, gemm_m, gemm_n, gemm_k, gemm_batch_size);

    // Backward: softmax_out_grad = qktv_out_grad * V^T (dx = dout * y^T)
    gemm_m = config->seq_len_r;
    gemm_n = config->m_size;
    gemm_k = config->key_dim;

    T* softmax_out_grad_ptr = softmax_out_grad->data<T>();
    ComputeBatchedGEMM(qktv_out_grad_ptr, v_ptr, softmax_out_grad_ptr, false,
                       true, gemm_m, gemm_n, gemm_k, gemm_batch_size);

    Tensor* qk_out_grad = config->GetQKOutGrad(dev_ctx_);
    ComputeBiasMaskSoftmaxBackward(softmax_out_grad, softmax_out, src_mask_grad,
                                   qk_out_grad, nonbatched_bias_grad);

    // Forward: qk_out = BatchedGEMM(Q, K^T)
    // Backward: k_grad = BatchedGEMM(qk_out_grad^T, Q) (dy = dout^t * x)
    gemm_m = config->m_size;
    gemm_n = config->key_dim;
    gemm_k = config->seq_len_r;
    T alpha = static_cast<T>(1.0 / sqrt(config->key_dim));

    T* qk_out_grad_ptr = qk_out_grad->data<T>();
    ComputeBatchedGEMM(qk_out_grad_ptr, q_ptr, k_grad_ptr, true, false, gemm_m,
                       gemm_n, gemm_k, gemm_batch_size, alpha);

    // Backward: q_grad = BatchedGEMM(qk_out_grad, K) (dx = dout * y)
    gemm_m = config->seq_len_r;
    gemm_n = config->key_dim;
    gemm_k = config->m_size;
    ComputeBatchedGEMM(qk_out_grad_ptr, k_ptr, q_grad_ptr, false, false, gemm_m,
                       gemm_n, gemm_k, gemm_batch_size, alpha);

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

  // [batch_size, seq_len_m, seq_len_r, 3, num_heads, key_dim] ->
  //         [3, batch_size, seq_len_m, num_heads, seq_len_r, key_dim]
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
      phi::funcs::BroadcastKernel<ElementwiseType::kTernary, T, T>(
          dev_ctx_, ins, &outs, -1, TernaryAddFunctor<T>());
    } else {
      std::vector<const Tensor*> ins = {qk_out, src_mask};
      std::vector<Tensor*> outs = {qk_out};
      phi::funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
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
  void ComputeBatchedGEMM(const T* a_ptr, const T* b_ptr, T* c_ptr,
                          bool trans_a, bool trans_b, int64_t m, int64_t n,
                          int64_t k, int64_t batch_size,
                          T alpha = static_cast<T>(1.0),
                          T beta = static_cast<T>(0.0)) {
    CBLAS_TRANSPOSE cblas_trans_a = trans_a ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE cblas_trans_b = trans_b ? CblasTrans : CblasNoTrans;
    int64_t stride_a = m * k;
    int64_t stride_b = k * n;

    auto blas = phi::funcs::GetBlas<platform::CUDADeviceContext, T>(dev_ctx_);
    blas.BatchedGEMM(cblas_trans_a, cblas_trans_b, m, n, k, alpha, a_ptr, b_ptr,
                     beta, c_ptr, batch_size, stride_a, stride_b);
  }

  const platform::CUDADeviceContext& dev_ctx_;
  bool merge_qkv_;
};

}  // namespace operators
}  // namespace paddle
