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
    gate_out_dims = {batch_size, seq_len_m, seq_len_r, num_heads, key_dim};
  }

  int64_t GetQuerySize() const {
    return batch_size * seq_len_m * seq_len_r * num_heads * key_dim;
  }

  Tensor* GetQKVOut(const platform::CUDADeviceContext& dev_ctx) {
    if (!qkv_out.IsInitialized()) {
      qkv_out.Resize(qkv_out_dims);
      qkv_out.mutable_data<T>(dev_ctx.GetPlace());
      VLOG(4) << "qkv_out: " << MemoryDebugString(qkv_out);
    }
    return &qkv_out;
  }

  Tensor* GetQueryOut(const platform::CUDADeviceContext& dev_ctx) {
    if (!query_out.IsInitialized()) {
      query_out.Resize(q_out_dims);
      query_out.mutable_data<T>(dev_ctx.GetPlace());
      VLOG(4) << "query_out: " << MemoryDebugString(query_out);
    }
    return &query_out;
  }

  Tensor* GetKeyOut(const platform::CUDADeviceContext& dev_ctx) {
    if (!key_out.IsInitialized()) {
      key_out.Resize(kv_out_dims);
      key_out.mutable_data<T>(dev_ctx.GetPlace());
      VLOG(4) << "key_out: " << MemoryDebugString(key_out);
    }
    return &key_out;
  }

  Tensor* GetValueOut(const platform::CUDADeviceContext& dev_ctx) {
    if (!value_out.IsInitialized()) {
      value_out.Resize(kv_out_dims);
      value_out.mutable_data<T>(dev_ctx.GetPlace());
      VLOG(4) << "value_out: " << MemoryDebugString(value_out);
    }
    return &value_out;
  }

  Tensor* GetQKOut(const platform::CUDADeviceContext& dev_ctx,
                   Tensor* softmax_out) {
    // softmax_dim = qk_out_dim[-1] = qk_out_dim[rank - 1]
    int softmax_dim = m_size;
    if (!softmax_out || phi::UseCudnnSoftmax<T>(dev_ctx, softmax_dim, true)) {
      // Not sure whether cudnn softmax can execute inplace.
      if (!qkv_out.IsInitialized()) {
        qk_out.Resize(qk_out_dims);
        qk_out.mutable_data<T>(dev_ctx.GetPlace());
        VLOG(4) << "qk_out: " << MemoryDebugString(qk_out);
      }
      return &qk_out;
    } else {
      return softmax_out;
    }
  }

  void ClearQKVOut() {
    if (qkv_out.IsInitialized()) {
      qkv_out.clear();
    }
  }

  void ClearQKOut() {
    if (qk_out.IsInitialized()) {
      qk_out.clear();
    }
  }

 protected:
  Tensor qkv_out;
  // QKV is not merged
  Tensor query_out;
  Tensor key_out;
  Tensor value_out;
  // qk_out = BatchedGEMM(Q, K^T)
  // qk_out: shape=[batch_size, seq_len_m, num_heads, seq_len_r, m_size]
  // softmax_out = softmax(qk_out + nonbatched_bias + src_mask)
  // The shape of qk_out, softmax_out is the same, thus can be called inplace.
  Tensor qk_out;
};

template <typename T>
struct GateAttentionGradConfig : public GateAttentionConfig<T> {
 public:
  GateAttentionGradConfig(const Tensor* query, const Tensor* key,
                          const Tensor* query_weight, const Tensor* qkv_weight,
                          bool merge_qkv)
      : GateAttentionConfig<T>(query, key, query_weight, qkv_weight,
                               merge_qkv) {}

  Tensor* GetQKVOutGrad(const platform::CUDADeviceContext& dev_ctx) {
    if (!qkv_out_grad.IsInitialized()) {
      qkv_out_grad.Resize(this->qkv_out_dims);
      qkv_out_grad.mutable_data<T>(dev_ctx.GetPlace());
      VLOG(4) << "qkv_out_grad: " << MemoryDebugString(qkv_out_grad);
    }
    return &qkv_out_grad;
  }

  Tensor* GetQueryOutGrad(const platform::CUDADeviceContext& dev_ctx) {
    if (!query_out_grad.IsInitialized()) {
      query_out_grad.Resize(this->q_out_dims);
      query_out_grad.mutable_data<T>(dev_ctx.GetPlace());
      VLOG(4) << "query_out_grad: " << MemoryDebugString(query_out_grad);
    }
    return &query_out_grad;
  }

  Tensor* GetKeyOutGrad(const platform::CUDADeviceContext& dev_ctx) {
    if (!key_out_grad.IsInitialized()) {
      key_out_grad.Resize(this->kv_out_dims);
      key_out_grad.mutable_data<T>(dev_ctx.GetPlace());
      VLOG(4) << "key_out_grad: " << MemoryDebugString(key_out_grad);
    }
    return &key_out_grad;
  }

  Tensor* GetValueOutGrad(const platform::CUDADeviceContext& dev_ctx) {
    if (!value_out_grad.IsInitialized()) {
      value_out_grad.Resize(this->kv_out_dims);
      value_out_grad.mutable_data<T>(dev_ctx.GetPlace());
      VLOG(4) << "value_out_grad: " << MemoryDebugString(value_out_grad);
    }
    return &value_out_grad;
  }

  Tensor* GetQKOutGrad(const platform::CUDADeviceContext& dev_ctx,
                       Tensor* softmax_out_grad) {
    // softmax_dim = qk_out_dim[-1] = qk_out_dim[rank - 1]
    int softmax_dim = this->m_size;
    if (!softmax_out_grad ||
        phi::UseCudnnSoftmax<T>(dev_ctx, softmax_dim, true)) {
      if (!qk_out_grad.IsInitialized()) {
        qk_out_grad.Resize(this->qk_out_dims);
        qk_out_grad.mutable_data<T>(dev_ctx.GetPlace());
        VLOG(4) << "qk_out_grad: " << MemoryDebugString(qk_out_grad);
      }
      return &qk_out_grad;
    } else {
      return softmax_out_grad;
    }
  }

 protected:
  Tensor qkv_out_grad;
  Tensor query_out_grad;
  Tensor key_out_grad;
  Tensor value_out_grad;
  Tensor qk_out_grad;
};

template <typename T>
class FMHAGateRef {
 public:
  FMHAGateRef(const platform::CUDADeviceContext& dev_ctx, bool merge_qkv)
      : dev_ctx_(dev_ctx), merge_qkv_(merge_qkv) {}

  void ComputeForward(const Tensor* nonbatched_bias, const Tensor* src_mask,
                      Tensor* q_transpose_out, Tensor* k_transpose_out,
                      Tensor* v_transpose_out, Tensor* qkv_transpose_out,
                      Tensor* softmax_out, Tensor* fmha_out,
                      GateAttentionConfig<T>* config) {
    T* q_ptr = nullptr;
    T* k_ptr = nullptr;
    T* v_ptr = nullptr;
    if (merge_qkv_) {
      // qkv_transpose_out = transpose(qkv_out)
      PADDLE_ENFORCE_NOT_NULL(qkv_transpose_out);

      Tensor* qkv_out = config->GetQKVOut(dev_ctx_);
      ComputeQKVTransposeForward(*qkv_out, qkv_transpose_out);
      config->ClearQKVOut();

      // q_size == k_size
      int64_t q_size = config->GetQuerySize();
      q_ptr = qkv_transpose_out->data<T>();
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + q_size;
    } else {
      PADDLE_ENFORCE_NOT_NULL(q_transpose_out);
      PADDLE_ENFORCE_NOT_NULL(k_transpose_out);
      PADDLE_ENFORCE_NOT_NULL(v_transpose_out);

      Tensor* query_out = config->GetQueryOut(dev_ctx_);
      Tensor* key_out = config->GetKeyOut(dev_ctx_);
      Tensor* value_out = config->GetValueOut(dev_ctx_);
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
    Tensor qktv_out;
    qktv_out.Resize(config->qktv_out_dims);
    T* qktv_out_ptr = qktv_out.mutable_data<T>(dev_ctx_.GetPlace());

    gemm_m = config->seq_len_r;
    gemm_n = config->key_dim;
    gemm_k = config->m_size;

    T* softmax_out_ptr = softmax_out->data<T>();
    ComputeBatchedGEMM(softmax_out_ptr, v_ptr, qktv_out_ptr, false, false,
                       gemm_m, gemm_n, gemm_k, gemm_batch_size);

    // fmha_out = transpose(qktv_out)
    ComputeQKTVTransposeForward(qktv_out, fmha_out);
  }

  void ComputeBackward(const Tensor* q_transpose_out,
                       const Tensor* k_transpose_out,
                       const Tensor* v_transpose_out,
                       const Tensor* qkv_transpose_out,
                       const Tensor* softmax_out, const Tensor* fmha_out_grad,
                       Tensor* src_mask_grad, Tensor* nonbatched_bias_grad,
                       GateAttentionGradConfig<T>* config) {
    const T* q_ptr = nullptr;
    const T* k_ptr = nullptr;
    const T* v_ptr = nullptr;

    T* q_grad_ptr = nullptr;
    T* k_grad_ptr = nullptr;
    T* v_grad_ptr = nullptr;

    Tensor q_transpose_out_grad;
    Tensor k_transpose_out_grad;
    Tensor v_transpose_out_grad;
    Tensor qkv_transpose_out_grad;
    if (merge_qkv_) {
      PADDLE_ENFORCE_NOT_NULL(qkv_transpose_out);

      int64_t q_size = config->GetQuerySize();
      q_ptr = qkv_transpose_out->data<T>();
      k_ptr = q_ptr + q_size;
      v_ptr = k_ptr + q_size;

      qkv_transpose_out_grad.Resize(config->qkv_transpose_out_dims);

      q_grad_ptr = qkv_transpose_out_grad.mutable_data<T>(dev_ctx_.GetPlace());
      k_grad_ptr = q_grad_ptr + q_size;
      v_grad_ptr = k_grad_ptr + q_size;
    } else {
      PADDLE_ENFORCE_NOT_NULL(q_transpose_out);
      PADDLE_ENFORCE_NOT_NULL(k_transpose_out);
      PADDLE_ENFORCE_NOT_NULL(v_transpose_out);

      q_ptr = q_transpose_out->data<T>();
      k_ptr = k_transpose_out->data<T>();
      v_ptr = v_transpose_out->data<T>();

      q_transpose_out_grad.Resize(config->q_transpose_out_dims);
      k_transpose_out_grad.Resize(config->kv_transpose_out_dims);
      v_transpose_out_grad.Resize(config->kv_transpose_out_dims);

      q_grad_ptr = q_transpose_out_grad.mutable_data<T>(dev_ctx_.GetPlace());
      k_grad_ptr = k_transpose_out_grad.mutable_data<T>(dev_ctx_.GetPlace());
      v_grad_ptr = v_transpose_out_grad.mutable_data<T>(dev_ctx_.GetPlace());
    }

    Tensor softmax_out_grad;
    softmax_out_grad.Resize(config->softmax_out_dims);
    softmax_out_grad.mutable_data<T>(dev_ctx_.GetPlace());

    int64_t gemm_batch_size =
        config->batch_size * config->seq_len_m * config->num_heads;
    {
      // Forward: fmha_out = transpose(qktv_out)
      Tensor qktv_out_grad;
      qktv_out_grad.Resize(config->qktv_out_dims);
      T* qktv_out_grad_ptr = qktv_out_grad.mutable_data<T>(dev_ctx_.GetPlace());
      ComputeQKTVTransposeBackward(*fmha_out_grad, &qktv_out_grad);

      // Forward: qktv_out = BatchedGEMM(softmax_out, V)
      // Backward:
      //  V_grad = BatchedGEMM(softmax_out^T, qktv_out_grad) (dy = x^T * dout)
      int64_t gemm_m = config->m_size;
      int64_t gemm_n = config->key_dim;
      int64_t gemm_k = config->seq_len_r;

      const T* softmax_out_ptr = softmax_out->data<T>();
      ComputeBatchedGEMM(softmax_out_ptr, qktv_out_grad_ptr, v_grad_ptr, true,
                         false, gemm_m, gemm_n, gemm_k, gemm_batch_size);

      // Backward: softmax_out_grad = qktv_out_grad * V^T (dx = dout * y^T)
      gemm_m = config->seq_len_r;
      gemm_n = config->m_size;
      gemm_k = config->key_dim;

      T* softmax_out_grad_ptr = softmax_out_grad.data<T>();
      ComputeBatchedGEMM(qktv_out_grad_ptr, v_ptr, softmax_out_grad_ptr, false,
                         true, gemm_m, gemm_n, gemm_k, gemm_batch_size);
    }

    Tensor* qk_out_grad = config->GetQKOutGrad(dev_ctx_, &softmax_out_grad);
    ComputeBiasMaskSoftmaxBackward(&softmax_out_grad, softmax_out,
                                   src_mask_grad, qk_out_grad,
                                   nonbatched_bias_grad);

    // Forward: qk_out = BatchedGEMM(Q, K^T)
    // Backward: k_grad = BatchedGEMM(qk_out_grad^T, Q) (dy = dout^t * x)
    int64_t gemm_m = config->m_size;
    int64_t gemm_n = config->key_dim;
    int64_t gemm_k = config->seq_len_r;
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
      Tensor* qkv_out_grad = config->GetQKVOutGrad(dev_ctx_);
      ComputeQKVTransposeBackward(qkv_transpose_out_grad, qkv_out_grad);
    } else {
      Tensor* q_out_grad = config->GetQueryOutGrad(dev_ctx_);
      Tensor* k_out_grad = config->GetKeyOutGrad(dev_ctx_);
      Tensor* v_out_grad = config->GetValueOutGrad(dev_ctx_);
      ComputeQKVTransposeBackward(q_transpose_out_grad, k_transpose_out_grad,
                                  v_transpose_out_grad, q_out_grad, k_out_grad,
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
