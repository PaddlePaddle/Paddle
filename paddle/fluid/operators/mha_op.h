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
    const Tensor* qo_slen = context.Input<Tensor>("QO_Seqlen");
    const Tensor* kv_slen = context.Input<Tensor>("KV_Seqlen");

    Tensor* o = context.Output<Tensor>("O");

    int batch_size = q->dims()[0];

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

    auto dtype = platform::CudnnDataType<T>::type;
    auto comp_prec =
        dtype == CUDNN_DATA_DOUBLE ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;

    auto cudnn_handle = dev_ctx.cudnn_handle();

    // TODO(Ming Huang): Need to come out a way to pass related variables from
    // FWD to BWD.
    std::string key = "ajskdlf";
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
        CUDNN_ATTN_QUERYMAP_ALL_TO_ONE, attn_heads, attn_sm_scaler, dtype,
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

    std::vector<int> qo_slen_host(qo_slen->dims()[0]);
    std::vector<int> kv_slen_host(kv_slen->dims()[0]);

    // TODO(rewang): use memory::Copy
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(qo_slen_host.data(),
                   reinterpret_cast<const void*>(qo_slen->data<int>()),
                   qo_slen->dims()[0] * sizeof(int), cudaMemcpyDeviceToHost));

    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(kv_slen_host.data(),
                   reinterpret_cast<const void*>(kv_slen->data<int>()),
                   kv_slen->dims()[0] * sizeof(int), cudaMemcpyDeviceToHost));

    cudnnSeqDataAxis_t
        axes[CUDNN_SEQDATA_DIM_COUNT];  // [Batch, Beam, Seq, Vec]
    axes[0] = CUDNN_SEQDATA_BATCH_DIM;
    axes[1] = CUDNN_SEQDATA_BEAM_DIM;
    axes[2] = CUDNN_SEQDATA_TIME_DIM;
    axes[3] = CUDNN_SEQDATA_VECT_DIM;

    int dimA[CUDNN_SEQDATA_DIM_COUNT];
    dimA[CUDNN_SEQDATA_VECT_DIM] = q->dims()[3];
    dimA[CUDNN_SEQDATA_TIME_DIM] = q->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = q->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = q->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).q_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        qo_slen_host.data(), nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = k->dims()[3];
    dimA[CUDNN_SEQDATA_TIME_DIM] = k->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = k->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = k->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).k_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        kv_slen_host.data(), nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = v->dims()[3];
    dimA[CUDNN_SEQDATA_TIME_DIM] = v->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = v->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = v->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).v_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        kv_slen_host.data(), nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = o->dims()[3];
    dimA[CUDNN_SEQDATA_TIME_DIM] = o->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = o->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = o->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).o_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        qo_slen_host.data(), nullptr));

    std::vector<int> attn_low_windows =
        context.Attr<std::vector<int>>("attn_low_windows");
    std::vector<int> attn_high_windows =
        context.Attr<std::vector<int>>("attn_high_windows");

    o->mutable_data<T>(context.GetPlace());
    const T* q_data = q->data<T>();
    const T* k_data = k->data<T>();
    const T* v_data = v->data<T>();
    const T* w_data = w->data<T>();

    T* o_data = o->data<T>();
    const T* residuals = nullptr;

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnMultiHeadAttnForward(
        cudnn_handle, MHASingleton::Instance().Data(key).attn_desc, -1,
        attn_low_windows.data(), attn_high_windows.data(), qo_slen->data<int>(),
        kv_slen->data<int>(), MHASingleton::Instance().Data(key).q_desc, q_data,
        residuals, MHASingleton::Instance().Data(key).k_desc, k_data,
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
    const Tensor* qo_slen = context.Input<Tensor>("QO_Seqlen");
    const Tensor* kv_slen = context.Input<Tensor>("KV_Seqlen");

    Tensor* dq = context.Output<Tensor>(framework::GradVarName("Q"));
    Tensor* dk = context.Output<Tensor>(framework::GradVarName("K"));
    Tensor* dv = context.Output<Tensor>(framework::GradVarName("V"));
    Tensor* dw = context.Output<Tensor>(framework::GradVarName("W"));

    auto cudnn_handle = dev_ctx.cudnn_handle();

    // TODO(Ming Huang): Need to come out a way to pass related variables from
    // FWD to BWD.
    std::string key = "ajskdlf";

    std::vector<int> attn_low_windows =
        context.Attr<std::vector<int>>("attn_low_windows");
    std::vector<int> attn_high_windows =
        context.Attr<std::vector<int>>("attn_high_windows");

    const T* dout_data = dout->data<T>();
    const T* q_data = q->data<T>();
    const T* k_data = k->data<T>();
    const T* v_data = v->data<T>();
    const T* w_data = w->data<T>();

    dq->mutable_data<T>(context.GetPlace());
    dk->mutable_data<T>(context.GetPlace());
    dv->mutable_data<T>(context.GetPlace());
    dw->mutable_data<T>(context.GetPlace());

    T* dq_data = dq->data<T>();
    T* dk_data = dk->data<T>();
    T* dv_data = dv->data<T>();
    T* dw_data = dw->data<T>();

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnMultiHeadAttnBackwardData(
            cudnn_handle, MHASingleton::Instance().Data(key).attn_desc,
            attn_low_windows.data(), attn_high_windows.data(),
            qo_slen->data<int>(), kv_slen->data<int>(),
            MHASingleton::Instance().Data(key).o_desc, dout_data,
            MHASingleton::Instance().Data(key).q_desc, dq_data, q_data,
            MHASingleton::Instance().Data(key).k_desc, dk_data, k_data,
            MHASingleton::Instance().Data(key).v_desc, dv_data, v_data,
            MHASingleton::Instance().Data(key).weights_size, w_data,
            MHASingleton::Instance().Data(key).workspace_size,
            MHASingleton::Instance().Data(key).workspace->ptr(),
            MHASingleton::Instance().Data(key).reserve_size,
            MHASingleton::Instance().Data(key).reserve_space->ptr()));
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
};

}  // namespace operators
}  // namespace paddle
