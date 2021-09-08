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
  cudnnDropoutDescriptor_t attn_dropout_desc = nullptr;
  cudnnDropoutDescriptor_t post_dropout_desc = nullptr;
  cudnnSeqDataDescriptor_t q_desc = nullptr;
  cudnnSeqDataDescriptor_t k_desc = nullptr;
  cudnnSeqDataDescriptor_t v_desc = nullptr;
  cudnnSeqDataDescriptor_t o_desc = nullptr;
  void* workspace = nullptr;
  void* reserve_space = nullptr;
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

    std::string key = "ajskdlf";

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnCreateAttnDescriptor(
        &MHASingleton::Instance().Data(key).attn_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnCreateSeqDataDescriptor(
        &MHASingleton::Instance().Data(key).q_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnCreateSeqDataDescriptor(
        &MHASingleton::Instance().Data(key).k_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnCreateSeqDataDescriptor(
        &MHASingleton::Instance().Data(key).v_desc));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnCreateSeqDataDescriptor(
        &MHASingleton::Instance().Data(key).o_desc));

    // Setup Attention Dropout
    if (attn_dropout_rate > 0.0) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cudnnCreateDropoutDescriptor(
              &MHASingleton::Instance().Data(key).attn_dropout_desc));

      size_t dropout_buf_size;
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnDropoutGetStatesSize(
          cudnn_handle, &dropout_buf_size));

      auto dropout_buf = memory::Alloc(dev_ctx, dropout_buf_size);
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetDropoutDescriptor(
          MHASingleton::Instance().Data(key).attn_dropout_desc, cudnn_handle,
          attn_dropout_rate, static_cast<void*>(dropout_buf->ptr()),
          dropout_buf_size, 0));
    }

    // Setup Attention Desc
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetAttnDescriptor(
        MHASingleton::Instance().Data(key).attn_desc,
        CUDNN_ATTN_QUERYMAP_ALL_TO_ONE, attn_heads, attn_sm_scaler, dtype,
        comp_prec, CUDNN_DEFAULT_MATH,
        MHASingleton::Instance().Data(key).attn_dropout_desc,
        MHASingleton::Instance().Data(key).post_dropout_desc, attn_vec_size,
        attn_vec_size, attn_vec_size, attn_q_proj_size, attn_k_proj_size,
        attn_v_proj_size, attn_o_proj_size, attn_max_qo_seq_len,
        attn_max_kv_seq_len, batch_size, attn_beam_size));

    size_t weights_size, wkspace_szie, reserve_size;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetMultiHeadAttnBuffers(
        cudnn_handle, MHASingleton::Instance().Data(key).attn_desc,
        &weights_size, &wkspace_szie, &reserve_size));
    auto weight_buf = memory::Alloc(dev_ctx, weights_size);
    auto wkspace_buf = memory::Alloc(dev_ctx, wkspace_szie);
    auto reserve_buf = memory::Alloc(dev_ctx, reserve_size);

    std::vector<int> q_seq_size_arr =
        context.Attr<std::vector<int>>("Q_seq_size_arr");
    std::vector<int> k_seq_size_arr =
        context.Attr<std::vector<int>>("K_seq_size_arr");

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
        q_seq_size_arr.data(), nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = k->dims()[3];
    dimA[CUDNN_SEQDATA_TIME_DIM] = k->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = k->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = k->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).k_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        k_seq_size_arr.data(), nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = v->dims()[3];
    dimA[CUDNN_SEQDATA_TIME_DIM] = v->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = v->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = v->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).v_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        k_seq_size_arr.data(), nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = o->dims()[3];
    dimA[CUDNN_SEQDATA_TIME_DIM] = o->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = o->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = o->dims()[0];
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).o_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        q_seq_size_arr.data(), nullptr));

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
    auto q_seq_size_dev_buf =
        memory::Alloc(dev_ctx, q_seq_size_arr.size() * sizeof(int));
    int* q_seq_size_dev_ptr = static_cast<int*>(q_seq_size_dev_buf->ptr());
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
        q_seq_size_dev_ptr, q_seq_size_arr.data(),
        q_seq_size_arr.size() * sizeof(int), cudaMemcpyHostToDevice));
    auto k_seq_size_dev_buf =
        memory::Alloc(dev_ctx, k_seq_size_arr.size() * sizeof(int));
    int* k_seq_size_dev_ptr = static_cast<int*>(k_seq_size_dev_buf->ptr());
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
        k_seq_size_dev_ptr, k_seq_size_arr.data(),
        k_seq_size_arr.size() * sizeof(int), cudaMemcpyHostToDevice));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnMultiHeadAttnForward(
        cudnn_handle, MHASingleton::Instance().Data(key).attn_desc, -1,
        attn_low_windows.data(), attn_high_windows.data(), q_seq_size_dev_ptr,
        k_seq_size_dev_ptr, MHASingleton::Instance().Data(key).q_desc, q_data,
        residuals, MHASingleton::Instance().Data(key).k_desc, k_data,
        MHASingleton::Instance().Data(key).v_desc, v_data,
        MHASingleton::Instance().Data(key).o_desc, o_data, weights_size, w_data,
        wkspace_szie, static_cast<void*>(wkspace_buf->ptr()), reserve_size,
        static_cast<void*>(reserve_buf->ptr())));
  }
};

template <typename DeviceContext, typename T>
class MHAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
