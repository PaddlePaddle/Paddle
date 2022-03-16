/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Corporation. All rights reserved.

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
#include "paddle/fluid/platform/device/gpu/cuda/cudnn_helper.h"

namespace paddle {
namespace operators {

class MHAMetaData {
 public:
  cudnnAttnDescriptor_t attn_desc = nullptr;
  cudnnSeqDataDescriptor_t query_desc = nullptr;
  cudnnSeqDataDescriptor_t key_desc = nullptr;
  cudnnSeqDataDescriptor_t value_desc = nullptr;
  cudnnSeqDataDescriptor_t output_desc = nullptr;

  memory::allocation::AllocationPtr workspace = nullptr;
  memory::allocation::AllocationPtr reserve_space = nullptr;

  size_t weight_size = 0;
  size_t workspace_size = 0;
  size_t reserve_size = 0;

  MHAMetaData() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateAttnDescriptor(&attn_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateSeqDataDescriptor(&query_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateSeqDataDescriptor(&key_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateSeqDataDescriptor(&value_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnCreateSeqDataDescriptor(&output_desc));
  }

  ~MHAMetaData() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroyAttnDescriptor(attn_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroySeqDataDescriptor(query_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroySeqDataDescriptor(key_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroySeqDataDescriptor(value_desc));
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnDestroySeqDataDescriptor(output_desc));
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
class cuDNNMHAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    const Tensor* query = context.Input<Tensor>("query");
    const Tensor* key = context.Input<Tensor>("key");
    const Tensor* value = context.Input<Tensor>("value");
    const Tensor* weight = context.Input<Tensor>("weight");
    const Tensor* residual = context.Input<Tensor>("residual");

    Tensor* output = context.Output<Tensor>("output");

    int batch_size = query->dims()[0];
    int seqlen = query->dims()[1];

    bool is_training = context.Attr<bool>("is_training");
    bool enable_bias = context.Attr<bool>("enable_bias");

    float pre_dropout_rate = context.Attr<float>("pre_dropout_rate");
    float post_dropout_rate = context.Attr<float>("post_dropout_rate");
    unsigned int dropout_seed =
        static_cast<unsigned int>(context.Attr<int>("seed"));
    int num_heads = context.Attr<int>("num_heads");
    double softmax_scaler =
        static_cast<double>(context.Attr<float>("softmax_scaler"));
    int embedding_size = context.Attr<int>("embedding_size");
    int query_proj_size = context.Attr<int>("query_proj_size");
    int key_proj_size = context.Attr<int>("key_proj_size");
    int value_proj_size = context.Attr<int>("value_proj_size");
    int output_proj_size = context.Attr<int>("output_proj_size");
    int max_qo_seqlen = context.Attr<int>("max_qo_seqlen");
    int max_kv_seqlen = context.Attr<int>("max_kv_seqlen");
    // int attn_beam_size = context.Attr<int>("attn_beam_size");
    int beam_size = 1;  // not support beam_dim currently.

    // TODO(Ming Huang): Work around now is using user-defined key
    const std::string cache_key = context.Attr<std::string>("cache_key");

    const Tensor* qo_kv_seqlen = context.Input<Tensor>("qo_kv_seqlen");
    const int* qo_kv_seqlen_data = qo_kv_seqlen->data<int>();
    const Tensor* qo_kv_seqlen_host =
        context.Input<Tensor>("qo_kv_seqlen_host");
    int* qo_kv_seqlen_data_host;
    memory::allocation::AllocationPtr qo_kv_seqlen_allc_ptr = nullptr;
    if (qo_kv_seqlen_host &&
        (platform::is_cpu_place(qo_kv_seqlen_host->place()) ||
         platform::is_cuda_pinned_place(qo_kv_seqlen_host->place()))) {
      qo_kv_seqlen_data_host = const_cast<int*>(qo_kv_seqlen_host->data<int>());
    } else {
      LOG(WARNING) << "[MHA Op]: qo_kv_seqlen_host is not given."
                   << " Copy from qo_kv_seqlen (CUDA place, Hurt performance).";

      platform::Place host_pinned_place = platform::CUDAPinnedPlace();
      size_t qo_kv_seqlen_size = qo_kv_seqlen->dims()[0] * sizeof(int);
      qo_kv_seqlen_allc_ptr =
          memory::Alloc(host_pinned_place, qo_kv_seqlen_size);
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaMemcpy(qo_kv_seqlen_allc_ptr->ptr(),
                     reinterpret_cast<const void*>(qo_kv_seqlen_data),
                     qo_kv_seqlen_size, cudaMemcpyDeviceToHost));
      qo_kv_seqlen_data_host =
          reinterpret_cast<int*>(qo_kv_seqlen_allc_ptr->ptr());
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
      size_t low_high_windows_size = 2 * seqlen * sizeof(int);
      low_high_windows_ptr =
          memory::Alloc(host_pinned_place, low_high_windows_size);

      if (low_high_windows_host &&
          platform::is_gpu_place(low_high_windows_host->place())) {
        LOG(WARNING)
            << "[MHA Op]: low_high_windows_host is given but in CUDA place."
            << " Copy from CUDA to CUDAPinned place (Hurt performance).";
        const int* low_high_windows_data = low_high_windows_host->data<int>();
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpy(low_high_windows_ptr->ptr(),
                       reinterpret_cast<const void*>(low_high_windows_data),
                       low_high_windows_size, cudaMemcpyDeviceToHost));
      } else {
        LOG(WARNING)
            << "[MHA Op]: low_high_windows_host is not given or not in CPU"
            << ", CUDAPinned and CUDAPinnded place. Gernereate 0s for low"
            << "and IMA_MAX for high.";
        int* lo_hi_ptr = reinterpret_cast<int*>(low_high_windows_ptr->ptr());
        for (int i = 0; i < seqlen; ++i) {
          lo_hi_ptr[i] = 0;
          lo_hi_ptr[i + seqlen] = INT_MAX;
        }
      }

      low_high_windows_data_host =
          reinterpret_cast<int*>(low_high_windows_ptr->ptr());
    }

    auto dtype = platform::CudnnDataType<T>::type;
    auto comp_prec =
        dtype == CUDNN_DATA_DOUBLE ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;

    auto cudnn_handle = dev_ctx.cudnn_handle();

    // Setup Pre Attention Dropout
    cudnnDropoutDescriptor_t pre_dropout_desc = nullptr;
    memory::allocation::AllocationPtr pre_dropout_buf_ptr = nullptr;
    if (pre_dropout_rate > 0.0) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnCreateDropoutDescriptor(&pre_dropout_desc));

      size_t dropout_buf_size;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnDropoutGetStatesSize(
          cudnn_handle, &dropout_buf_size));

      pre_dropout_buf_ptr = memory::Alloc(dev_ctx, dropout_buf_size);
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetDropoutDescriptor(
          pre_dropout_desc, cudnn_handle, pre_dropout_rate,
          static_cast<void*>(pre_dropout_buf_ptr->ptr()), dropout_buf_size,
          dropout_seed));
    }

    // Setup Post Attention Dropout
    cudnnDropoutDescriptor_t post_dropout_desc = nullptr;
    memory::allocation::AllocationPtr post_dropout_buf_ptr = nullptr;
    if (post_dropout_rate > 0.0) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnCreateDropoutDescriptor(&post_dropout_desc));

      size_t dropout_buf_size;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnDropoutGetStatesSize(
          cudnn_handle, &dropout_buf_size));

      post_dropout_buf_ptr = memory::Alloc(dev_ctx, dropout_buf_size);
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetDropoutDescriptor(
          post_dropout_desc, cudnn_handle, post_dropout_rate,
          static_cast<void*>(post_dropout_buf_ptr->ptr()), dropout_buf_size,
          dropout_seed));
    }

    // Setup Attention Desc
    auto attn_mode = enable_bias ? CUDNN_ATTN_ENABLE_PROJ_BIASES
                                 : CUDNN_ATTN_DISABLE_PROJ_BIASES;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetAttnDescriptor(
        MHASingleton::Instance().Data(cache_key).attn_desc, attn_mode,
        num_heads, softmax_scaler, dtype, comp_prec, CUDNN_DEFAULT_MATH,
        pre_dropout_desc, post_dropout_desc, embedding_size, embedding_size,
        embedding_size, query_proj_size, key_proj_size, value_proj_size,
        output_proj_size, max_qo_seqlen, max_kv_seqlen, batch_size, beam_size));

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnGetMultiHeadAttnBuffers(
        cudnn_handle, MHASingleton::Instance().Data(cache_key).attn_desc,
        &MHASingleton::Instance().Data(cache_key).weight_size,
        &MHASingleton::Instance().Data(cache_key).workspace_size,
        &MHASingleton::Instance().Data(cache_key).reserve_size));

    // TODO(rewang): ensure workspace size will not be increased
    if (MHASingleton::Instance().Data(cache_key).workspace == nullptr) {
      MHASingleton::Instance().Data(cache_key).workspace = memory::Alloc(
          dev_ctx, MHASingleton::Instance().Data(cache_key).workspace_size);
    }

    // TODO(rewang): ensure reserve_size size will not be increased
    if (is_training &&
        MHASingleton::Instance().Data(cache_key).reserve_space == nullptr) {
      MHASingleton::Instance().Data(cache_key).reserve_space = memory::Alloc(
          dev_ctx, MHASingleton::Instance().Data(cache_key).reserve_size);
    } else {
      MHASingleton::Instance().Data(cache_key).reserve_space == nullptr;
      MHASingleton::Instance().Data(cache_key).reserve_size == 0;
    }

    cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];
    axes[0] = CUDNN_SEQDATA_BATCH_DIM;
    axes[1] = CUDNN_SEQDATA_TIME_DIM;
    axes[2] = CUDNN_SEQDATA_BEAM_DIM;
    axes[3] = CUDNN_SEQDATA_VECT_DIM;

    int dimA[CUDNN_SEQDATA_DIM_COUNT];
    dimA[CUDNN_SEQDATA_VECT_DIM] = query->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = beam_size;
    dimA[CUDNN_SEQDATA_TIME_DIM] = query->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = query->dims()[0];
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(cache_key).query_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * beam_size,
        qo_kv_seqlen_data_host, nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = key->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = beam_size;
    dimA[CUDNN_SEQDATA_TIME_DIM] = key->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = key->dims()[0];
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(cache_key).key_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * beam_size,
        qo_kv_seqlen_data_host + batch_size, nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = value->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = beam_size;
    dimA[CUDNN_SEQDATA_TIME_DIM] = value->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = value->dims()[0];
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(cache_key).value_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * beam_size,
        qo_kv_seqlen_data_host + batch_size, nullptr));

    dimA[CUDNN_SEQDATA_VECT_DIM] = output->dims()[2];
    dimA[CUDNN_SEQDATA_BEAM_DIM] = beam_size;
    dimA[CUDNN_SEQDATA_TIME_DIM] = output->dims()[1];
    dimA[CUDNN_SEQDATA_BATCH_DIM] = output->dims()[0];
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(cache_key).output_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * beam_size,
        qo_kv_seqlen_data_host, nullptr));

    output->mutable_data<T>(context.GetPlace());
    const T* query_data = query->data<T>();
    const T* key_data = key->data<T>();
    const T* value_data = value->data<T>();
    const T* weight_data = weight->data<T>();
    const T* residual_data = nullptr;
    if (residual) {
      residual_data = residual->data<T>();
    }

    T* output_data = output->data<T>();

    void* rs_ptr = nullptr;
    if (is_training) {
      rs_ptr = MHASingleton::Instance().Data(cache_key).reserve_space->ptr();
    }

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cudnnMultiHeadAttnForward(
        cudnn_handle, MHASingleton::Instance().Data(cache_key).attn_desc, -1,
        low_high_windows_data_host, low_high_windows_data_host + seqlen,
        qo_kv_seqlen_data, qo_kv_seqlen_data + batch_size,
        MHASingleton::Instance().Data(cache_key).query_desc, query_data,
        residual_data, MHASingleton::Instance().Data(cache_key).key_desc,
        key_data, MHASingleton::Instance().Data(cache_key).value_desc,
        value_data, MHASingleton::Instance().Data(cache_key).output_desc,
        output_data, MHASingleton::Instance().Data(cache_key).weight_size,
        weight_data, MHASingleton::Instance().Data(cache_key).workspace_size,
        MHASingleton::Instance().Data(cache_key).workspace->ptr(),
        MHASingleton::Instance().Data(cache_key).reserve_size, rs_ptr));
  }
};

template <typename DeviceContext, typename T>
class cuDNNMHAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    const Tensor* doutput =
        context.Input<Tensor>(framework::GradVarName("output"));
    const Tensor* query = context.Input<Tensor>("query");
    const Tensor* key = context.Input<Tensor>("key");
    const Tensor* value = context.Input<Tensor>("value");
    const Tensor* weight = context.Input<Tensor>("weight");
    const Tensor* qo_kv_seqlen = context.Input<Tensor>("qo_kv_seqlen");

    Tensor* dquery = context.Output<Tensor>(framework::GradVarName("query"));
    Tensor* dkey = context.Output<Tensor>(framework::GradVarName("key"));
    Tensor* dvalue = context.Output<Tensor>(framework::GradVarName("value"));
    Tensor* dweight = context.Output<Tensor>(framework::GradVarName("weight"));
    Tensor* dresidual =
        context.Output<Tensor>(framework::GradVarName("residual"));

    auto cudnn_handle = dev_ctx.cudnn_handle();

    // TODO(Ming Huang): Work around now is using user-defined key
    const std::string cache_key = context.Attr<std::string>("cache_key");

    int seqlen = query->dims()[1];
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
      size_t low_high_windows_size = 2 * seqlen * sizeof(int);
      low_high_windows_ptr =
          memory::Alloc(host_pinned_place, low_high_windows_size);

      if (low_high_windows_host &&
          platform::is_gpu_place(low_high_windows_host->place())) {
        LOG(WARNING)
            << "[MHA Grad Op]: low_high_windows_host is given but in CUDA "
               "place."
            << " Copy from CUDA to CUDAPinned place (Hurt performance).";
        const int* low_high_windows_data = low_high_windows_host->data<int>();
        PADDLE_ENFORCE_GPU_SUCCESS(
            cudaMemcpy(low_high_windows_ptr->ptr(),
                       reinterpret_cast<const void*>(low_high_windows_data),
                       low_high_windows_size, cudaMemcpyDeviceToHost));
      } else {
        LOG(WARNING)
            << "[MHA Grad Op]: low_high_windows_host is not given or not in CPU"
            << ", CUDAPinned and CUDAPinnded place. Gernereate 0s for low"
            << "and IMA_MAX for high.";
        int* lo_hi_ptr = reinterpret_cast<int*>(low_high_windows_ptr->ptr());
        for (int i = 0; i < seqlen; ++i) {
          lo_hi_ptr[i] = 0;
          lo_hi_ptr[i + seqlen] = INT_MAX;
        }
      }

      low_high_windows_data_host =
          reinterpret_cast<int*>(low_high_windows_ptr->ptr());
    }

    const T* doutput_data = doutput->data<T>();
    const T* query_data = query->data<T>();
    const T* key_data = key->data<T>();
    const T* value_data = value->data<T>();
    const T* weight_data = weight->data<T>();

    if (dresidual) {
      dresidual->mutable_data<T>(context.GetPlace());
      T* dresidual_data = dresidual->data<T>();
      size_t cpy_size = static_cast<size_t>(phi::product(dresidual->dims()));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(dresidual_data, doutput_data,
                                                 cpy_size * sizeof(T),
                                                 cudaMemcpyDeviceToDevice));
    }

    // Note(Ming Huang): Due to cuDNN MHA need to call BwdData before
    // BwdWeight for correctly gradient computing, we need tempoary
    // memory buffers for DQ, DK and DV for calling BwdData when they.
    // are nullptrs.
    T* dquery_data;
    if (dquery) {
      dquery->mutable_data<T>(context.GetPlace());
      dquery_data = dquery->data<T>();
    } else {
      auto dquery_buf =
          memory::Alloc(dev_ctx, phi::product(query->dims()) * sizeof(T));
      dquery_data = static_cast<T*>(dquery_buf->ptr());
    }

    T* dkey_data;
    if (dkey) {
      dkey->mutable_data<T>(context.GetPlace());
      dkey_data = dkey->data<T>();
    } else {
      auto dkey_buf =
          memory::Alloc(dev_ctx, phi::product(key->dims()) * sizeof(T));
      dkey_data = static_cast<T*>(dkey_buf->ptr());
    }

    T* dvalue_data;
    if (dvalue) {
      dvalue->mutable_data<T>(context.GetPlace());
      dvalue_data = dvalue->data<T>();
    } else {
      auto dvalue_buf =
          memory::Alloc(dev_ctx, phi::product(value->dims()) * sizeof(T));
      dvalue_data = static_cast<T*>(dvalue_buf->ptr());
    }

    const int* qo_kv_seqlen_data = qo_kv_seqlen->data<int>();
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cudnnMultiHeadAttnBackwardData(
            cudnn_handle, MHASingleton::Instance().Data(cache_key).attn_desc,
            low_high_windows_data_host, low_high_windows_data_host + seqlen,
            qo_kv_seqlen_data, qo_kv_seqlen_data + query->dims()[0],
            MHASingleton::Instance().Data(cache_key).output_desc, doutput_data,
            MHASingleton::Instance().Data(cache_key).query_desc, dquery_data,
            query_data, MHASingleton::Instance().Data(cache_key).key_desc,
            dkey_data, key_data,
            MHASingleton::Instance().Data(cache_key).value_desc, dvalue_data,
            value_data, MHASingleton::Instance().Data(cache_key).weight_size,
            weight_data,
            MHASingleton::Instance().Data(cache_key).workspace_size,
            MHASingleton::Instance().Data(cache_key).workspace->ptr(),
            MHASingleton::Instance().Data(cache_key).reserve_size,
            MHASingleton::Instance().Data(cache_key).reserve_space->ptr()));

    if (dweight) {
      dweight->mutable_data<T>(context.GetPlace());
      T* dweight_data = dweight->data<T>();

      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cudnnMultiHeadAttnBackwardWeights(
              cudnn_handle, MHASingleton::Instance().Data(cache_key).attn_desc,
              CUDNN_WGRAD_MODE_SET,
              MHASingleton::Instance().Data(cache_key).query_desc, query_data,
              MHASingleton::Instance().Data(cache_key).key_desc, key_data,
              MHASingleton::Instance().Data(cache_key).value_desc, value_data,
              MHASingleton::Instance().Data(cache_key).output_desc,
              doutput_data,
              MHASingleton::Instance().Data(cache_key).weight_size, weight_data,
              dweight_data,
              MHASingleton::Instance().Data(cache_key).workspace_size,
              MHASingleton::Instance().Data(cache_key).workspace->ptr(),
              MHASingleton::Instance().Data(cache_key).reserve_size,
              MHASingleton::Instance().Data(cache_key).reserve_space->ptr()));
    }
  }
};

}  // namespace operators
}  // namespace paddle
