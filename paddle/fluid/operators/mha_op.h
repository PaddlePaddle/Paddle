/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2021 NVIDIA Corporation. All rights reserved.

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

#include <cudnn.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

class MHAMetaData {
 public:
  cudnnAttnDescriptor_t attn_desc = nullptr;
  cudnnSeqDataDescriptor_t q_desc = nullptr;
  cudnnSeqDataDescriptor_t k_desc = nullptr;
  cudnnSeqDataDescriptor_t v_desc = nullptr;
  cudnnSeqDataDescriptor_t o_desc = nullptr;
  cudnnHandle_t cudnn_handle = nullptr;

  memory::allocation::AllocationPtr workspace = nullptr;
  memory::allocation::AllocationPtr reserve_space = nullptr;

  size_t weights_size = 0;
  size_t workspace_size = 0;
  size_t reserve_size = 0;

  void SetCudnnHandle(cudnnHandle_t cudnn_handle) {
    cudnn_handle = cudnn_handle;
  }

  MHAMetaData() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateAttnDescriptor(&attn_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateSeqDataDescriptor(&q_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateSeqDataDescriptor(&k_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateSeqDataDescriptor(&v_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnCreateSeqDataDescriptor(&o_desc));
  }

  ~MHAMetaData() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroyAttnDescriptor(attn_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroySeqDataDescriptor(q_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroySeqDataDescriptor(k_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroySeqDataDescriptor(v_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnDestroySeqDataDescriptor(o_desc));
  }
};

class MHASingleton {
 public:
  static MHASingleton& Instance() {
    static MHASingleton instance;
    return instance;
  }

  MHASingleton(MHASingleton const&) = delete;
  void operator=(MHASingleton const&) = delete;

  MHAMetaData& Data(const std::string& str) { return map_[str]; }

 private:
  MHASingleton() {}
  std::unordered_map<std::string, MHAMetaData> map_;
};

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class MHAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    const Tensor* q = context.Input<Tensor>("Q");
    const Tensor* k = context.Input<Tensor>("K");
    const Tensor* v = context.Input<Tensor>("V");
    const Tensor* w = context.Input<Tensor>("W");

    Tensor* o = context.Output<Tensor>("O");

    int batch_size = q->dims()[0];
    int seq_len = q->dims()[1];

    float attn_dropout_rate = context.Attr<float>("attn_dropout_rate");
    int attn_heads = context.Attr<int>("attn_heads");
    double attn_sm_scaler =
        static_cast<double>(context.Attr<float>("attn_sm_scaler"));
    int attn_vec_size = context.Attr<int>("attn_vec_size");
    int attn_q_proj_size = context.Attr<int>("attn_q_proj_size");
    int attn_k_proj_size = context.Attr<int>("attn_k_proj_size");
    int attn_v_proj_size = context.Attr<int>("attn_v_proj_size");
    int attn_o_proj_size = context.Attr<int>("attn_o_proj_size");
    int attn_max_qo_seq_len = context.Attr<int>("attn_max_qo_seq_len");
    int attn_max_kv_seq_len = context.Attr<int>("attn_max_kv_seq_len");
    int attn_beam_size = context.Attr<int>("attn_beam_size");

    // TODO(Ming Huang): Work around now is using user-defined key
    const std::string key = context.Attr<std::string>("cache_key");

    const Tensor* qo_kv_slen = context.Input<Tensor>("QO_KV_Seqlen");
    const int* qo_kv_slen_data = qo_kv_slen->data<int>();
    const Tensor* qo_kv_slen_host = context.Input<Tensor>("QO_KV_Seqlen_host");
    int* qo_kv_slen_data_host;
    memory::allocation::AllocationPtr qo_kv_slen_allc_ptr = nullptr;
    if (qo_kv_slen_host &&
        (platform::is_cpu_place(qo_kv_slen_host->place()) ||
         platform::is_cuda_pinned_place(qo_kv_slen_host->place()))) {
      qo_kv_slen_data_host = const_cast<int*>(qo_kv_slen_host->data<int>());
    } else {
      VLOG(0) << "[MHA Op]: QO_KV_Seqlen_host is not given."
              << " Copy from QO_KV_Seqlen (CUDA place, Hurt performance).";

      platform::Place host_pinned_place = platform::CUDAPinnedPlace();
      size_t qkvo_seqlen_size = qo_kv_slen->dims()[0] * sizeof(int);
      qo_kv_slen_allc_ptr = memory::Alloc(host_pinned_place, qkvo_seqlen_size);
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaMemcpy(qo_kv_slen_allc_ptr->ptr(),
                     reinterpret_cast<const void*>(qo_kv_slen_data),
                     qkvo_seqlen_size, cudaMemcpyDeviceToHost));
      qo_kv_slen_data_host = reinterpret_cast<int*>(qo_kv_slen_allc_ptr->ptr());
    }

    const Tensor* low_high_windows_host =
        context.Input<Tensor>("low_high_windows_host");
    int* low_high_windows_data_host;
    memory::allocation::AllocationPtr low_high_windows_ptr = nullptr;
    if (low_high_windows_host &&
        (platform::is_cpu_place(low_high_windows_host->place()) ||
         platform::is_cuda_pinned_place(low_high_windows_host->place()))) {
      low_high_windows_data_host =
          const_cast<int*>(low_high_windows_host->data<int>());
    } else {
      platform::Place host_pinned_place = platform::CUDAPinnedPlace();
      size_t low_high_windows_size = 2 * seq_len * sizeof(int);
      low_high_windows_ptr =
          memory::Alloc(host_pinned_place, low_high_windows_size);

      if (low_high_windows_host &&
          platform::is_gpu_place(low_high_windows_host->place())) {
        VLOG(0) << "[MHA Op]: low_high_windows_host is given but in CUDA place."
                << " Copy from CUDA to CUDAPinned place (Hurt performance).";
        const int* low_high_windows_data = low_high_windows_host->data<int>();
        PADDLE_ENFORCE_CUDA_SUCCESS(
            cudaMemcpy(low_high_windows_ptr->ptr(),
                       reinterpret_cast<const void*>(low_high_windows_data),
                       low_high_windows_size, cudaMemcpyDeviceToHost));
      } else {
        VLOG(0) << "[MHA Op]: low_high_windows_host is not given or not in CPU"
                << ", CUDAPinned and CUDAPinnded place. Gernereate 0s for low "
                   "and IMA_MAX for high.";
        int* lo_hi_ptr = reinterpret_cast<int*>(low_high_windows_ptr->ptr());
        for (int i = 0; i < seq_len; ++i) {
          lo_hi_ptr[i] = 0;
          lo_hi_ptr[i + seq_len] = INT_MAX;
        }
      }

      low_high_windows_data_host =
          reinterpret_cast<int*>(low_high_windows_ptr->ptr());
    }

    auto dtype = platform::CudnnDataType<T>::type;
    auto comp_prec =
        dtype == CUDNN_DATA_DOUBLE ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;

    auto cudnn_handle = dev_ctx.cudnn_handle();
    MHASingleton::Instance().Data(key).SetCudnnHandle(cudnn_handle);

    cudnnDropoutDescriptor_t attn_dropout_desc = nullptr;
    cudnnDropoutDescriptor_t post_dropout_desc = nullptr;
    // Setup Attention Dropout
    if (attn_dropout_rate > 0.0) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnCreateDropoutDescriptor(&attn_dropout_desc));

      size_t dropout_buf_size;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnDropoutGetStatesSize(
          cudnn_handle, &dropout_buf_size));

      auto dropout_buf = memory::Alloc(dev_ctx, dropout_buf_size);
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetDropoutDescriptor(
          attn_dropout_desc, cudnn_handle, attn_dropout_rate,
          static_cast<void*>(dropout_buf->ptr()), dropout_buf_size, 0));
    }

    // Setup Attention Desc
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetAttnDescriptor(
        MHASingleton::Instance().Data(key).attn_desc,
        CUDNN_ATTN_ENABLE_PROJ_BIASES, attn_heads, attn_sm_scaler, dtype,
        comp_prec, CUDNN_DEFAULT_MATH, attn_dropout_desc, post_dropout_desc,
        attn_vec_size, attn_vec_size, attn_vec_size, attn_q_proj_size,
        attn_k_proj_size, attn_v_proj_size, attn_o_proj_size,
        attn_max_qo_seq_len, attn_max_kv_seq_len, batch_size, attn_beam_size));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetMultiHeadAttnBuffers(
        cudnn_handle, MHASingleton::Instance().Data(key).attn_desc,
        &MHASingleton::Instance().Data(key).weights_size,
        &MHASingleton::Instance().Data(key).workspace_size,
        &MHASingleton::Instance().Data(key).reserve_size));

    // TODO(rewang): ensure workspace size will not be increased
    if (MHASingleton::Instance().Data(key).workspace == nullptr) {
      MHASingleton::Instance().Data(key).workspace = memory::Alloc(
          dev_ctx, MHASingleton::Instance().Data(key).workspace_size);
    }

    // TODO(rewang): ensure reserve_size size will not be increased
    if (MHASingleton::Instance().Data(key).reserve_space == nullptr) {
      MHASingleton::Instance().Data(key).reserve_space = memory::Alloc(
          dev_ctx, MHASingleton::Instance().Data(key).reserve_size);
    }

    cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
    axes[0] = CUDNN_SEQDATA_BATCH_DIM;
    axes[1] = CUDNN_SEQDATA_TIME_DIM;
    axes[2] = CUDNN_SEQDATA_BEAM_DIM;
    axes[3] = CUDNN_SEQDATA_VECT_DIM;

    int dimA[CUDNN_SEQDATA_DIM_COUNT];
    dimA[CUDNN_SEQDATA_VECT_DIM] = q->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;  // not support beam_dim currently.
    dimA[CUDNN_SEQDATA_TIME_DIM] = q->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = q->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).q_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        qo_kv_slen_data_host, nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = k->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;  // not support beam_dim currently.
    dimA[CUDNN_SEQDATA_TIME_DIM] = k->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = k->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).k_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        qo_kv_slen_data_host + batch_size, nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = v->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;  // not support beam_dim currently.
    dimA[CUDNN_SEQDATA_TIME_DIM] = v->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = v->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).v_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        qo_kv_slen_data_host + batch_size, nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = o->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;  // not support beam_dim currently.
    dimA[CUDNN_SEQDATA_TIME_DIM] = o->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = o->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).o_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        qo_kv_slen_data_host, nullptr));

    o->mutable_data<T>(context.GetPlace());
    const T* q_data = q->data<T>();
    const T* k_data = k->data<T>();
    const T* v_data = v->data<T>();
    const T* w_data = w->data<T>();

    T* o_data = o->data<T>();
    const T* residuals = nullptr;

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnMultiHeadAttnForward(
        cudnn_handle, MHASingleton::Instance().Data(key).attn_desc, -1,
        low_high_windows_data_host, low_high_windows_data_host + seq_len,
        qo_kv_slen_data, qo_kv_slen_data + batch_size,
        MHASingleton::Instance().Data(key).q_desc, q_data, residuals,
        MHASingleton::Instance().Data(key).k_desc, k_data,
        MHASingleton::Instance().Data(key).v_desc, v_data,
        MHASingleton::Instance().Data(key).o_desc, o_data,
        MHASingleton::Instance().Data(key).weights_size, w_data,
        MHASingleton::Instance().Data(key).workspace_size,
        MHASingleton::Instance().Data(key).workspace->ptr(),
        MHASingleton::Instance().Data(key).reserve_size,
        MHASingleton::Instance().Data(key).reserve_space->ptr()));
  }
};

template <typename DeviceContext, typename T>
class MHAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    const Tensor* dout = context.Input<Tensor>(framework::GradVarName("O"));
    const Tensor* q = context.Input<Tensor>("Q");
    const Tensor* k = context.Input<Tensor>("K");
    const Tensor* v = context.Input<Tensor>("V");
    const Tensor* w = context.Input<Tensor>("W");
    const Tensor* qo_kv_slen = context.Input<Tensor>("QO_KV_Seqlen");

    Tensor* dq = context.Output<Tensor>(framework::GradVarName("Q"));
    Tensor* dk = context.Output<Tensor>(framework::GradVarName("K"));
    Tensor* dv = context.Output<Tensor>(framework::GradVarName("V"));
    Tensor* dw = context.Output<Tensor>(framework::GradVarName("W"));

    auto cudnn_handle = dev_ctx.cudnn_handle();

    // TODO(Ming Huang): Work around now is using user-defined key
    const std::string key = context.Attr<std::string>("cache_key");

    int seq_len = q->dims()[1];
    const Tensor* low_high_windows_host =
        context.Input<Tensor>("low_high_windows_host");
    int* low_high_windows_data_host;
    memory::allocation::AllocationPtr low_high_windows_ptr = nullptr;
    if (low_high_windows_host &&
        (platform::is_cpu_place(low_high_windows_host->place()) ||
         platform::is_cuda_pinned_place(low_high_windows_host->place()))) {
      low_high_windows_data_host =
          const_cast<int*>(low_high_windows_host->data<int>());
    } else {
      platform::Place host_pinned_place = platform::CUDAPinnedPlace();
      size_t low_high_windows_size = 2 * seq_len * sizeof(int);
      low_high_windows_ptr =
          memory::Alloc(host_pinned_place, low_high_windows_size);

      if (low_high_windows_host &&
          platform::is_gpu_place(low_high_windows_host->place())) {
        const int* low_high_windows_data = low_high_windows_host->data<int>();
        PADDLE_ENFORCE_CUDA_SUCCESS(
            cudaMemcpy(low_high_windows_ptr->ptr(),
                       reinterpret_cast<const void*>(low_high_windows_data),
                       low_high_windows_size, cudaMemcpyDeviceToHost));
      } else {
        int* lo_hi_ptr = reinterpret_cast<int*>(low_high_windows_ptr->ptr());
        for (int i = 0; i < seq_len; ++i) {
          lo_hi_ptr[i] = 0;
          lo_hi_ptr[i + seq_len] = INT_MAX;
        }
      }

      low_high_windows_data_host =
          reinterpret_cast<int*>(low_high_windows_ptr->ptr());
    }

    const T* dout_data = dout->data<T>();
    const T* q_data = q->data<T>();
    const T* k_data = k->data<T>();
    const T* v_data = v->data<T>();
    const T* w_data = w->data<T>();

    // Note(Ming Huang): Due to cuDNN MHA need to call BwdData before
    // BwdWeight for correctly gradient computing, we need tempoary
    // memory buffers for DQ, DK and DV for calling BwdData when they.
    // are nullptrs.
    T* dq_data;
    if (dq) {
      dq->mutable_data<T>(context.GetPlace());
      dq_data = dq->data<T>();
    } else {
      auto dq_buf =
          memory::Alloc(dev_ctx, framework::product(q->dims()) * sizeof(T));
      dq_data = static_cast<T*>(dq_buf->ptr());
    }

    T* dk_data;
    if (dk) {
      dk->mutable_data<T>(context.GetPlace());
      dk_data = dk->data<T>();
    } else {
      auto dk_buf =
          memory::Alloc(dev_ctx, framework::product(k->dims()) * sizeof(T));
      dk_data = static_cast<T*>(dk_buf->ptr());
    }

    T* dv_data;
    if (dv) {
      dv->mutable_data<T>(context.GetPlace());
      dv_data = dv->data<T>();
    } else {
      auto dv_buf =
          memory::Alloc(dev_ctx, framework::product(v->dims()) * sizeof(T));
      dv_data = static_cast<T*>(dv_buf->ptr());
    }

    const int* qo_kv_slen_data = qo_kv_slen->data<int>();
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnMultiHeadAttnBackwardData(
            cudnn_handle, MHASingleton::Instance().Data(key).attn_desc,
            low_high_windows_data_host, low_high_windows_data_host + seq_len,
            qo_kv_slen_data, qo_kv_slen_data + q->dims()[0],
            MHASingleton::Instance().Data(key).o_desc, dout_data,
            MHASingleton::Instance().Data(key).q_desc, dq_data, q_data,
            MHASingleton::Instance().Data(key).k_desc, dk_data, k_data,
            MHASingleton::Instance().Data(key).v_desc, dv_data, v_data,
            MHASingleton::Instance().Data(key).weights_size, w_data,
            MHASingleton::Instance().Data(key).workspace_size,
            MHASingleton::Instance().Data(key).workspace->ptr(),
            MHASingleton::Instance().Data(key).reserve_size,
            MHASingleton::Instance().Data(key).reserve_space->ptr()));

    if (dw) {
      dw->mutable_data<T>(context.GetPlace());
      T* dw_data = dw->data<T>();

      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnMultiHeadAttnBackwardWeights(
              cudnn_handle, MHASingleton::Instance().Data(key).attn_desc,
              CUDNN_WGRAD_MODE_SET, MHASingleton::Instance().Data(key).q_desc,
              q_data, MHASingleton::Instance().Data(key).k_desc, k_data,
              MHASingleton::Instance().Data(key).v_desc, v_data,
              MHASingleton::Instance().Data(key).o_desc, dout_data,
              MHASingleton::Instance().Data(key).weights_size, w_data, dw_data,
              MHASingleton::Instance().Data(key).workspace_size,
              MHASingleton::Instance().Data(key).workspace->ptr(),
              MHASingleton::Instance().Data(key).reserve_size,
              MHASingleton::Instance().Data(key).reserve_space->ptr()));
    }
  }
};

}  // namespace operators
}  // namespace paddle
