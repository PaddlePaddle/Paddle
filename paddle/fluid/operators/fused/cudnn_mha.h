/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace operators {

#if CUDNN_VERSION >= 8000
class MHAMetaData {
 public:
  cudnnAttnDescriptor_t attn_desc = nullptr;
  cudnnSeqDataDescriptor_t q_desc = nullptr;
  cudnnSeqDataDescriptor_t k_desc = nullptr;
  cudnnSeqDataDescriptor_t v_desc = nullptr;
  cudnnSeqDataDescriptor_t o_desc = nullptr;
  cudnnHandle_t cudnn_handle = nullptr;

  memory::allocation::AllocationPtr workspace = nullptr;
  // memory::allocation::AllocationPtr reserve_space = nullptr;

  size_t weights_size = 0;
  size_t workspace_size = 0;
  // size_t reserve_size = 0;

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

// template <typename DeviceContext, typename T>
// class MHAKernel : public framework::OpKernel<T> {
//  public:
//   void Compute(const framework::ExecutionContext& context) const override {

template <typename T>
void MHAFwKernel(const platform::CUDADeviceContext& dev_ctx, bool has_bias,
                 const Tensor* q, const Tensor* k, const Tensor* v,
                 const Tensor* w, const Tensor* qo_slen, const Tensor* kv_slen,
                 float attn_dropout_rate, int attn_heads, double attn_sm_scaler,
                 int attn_vec_size, int attn_q_proj_size, int attn_k_proj_size,
                 int attn_v_proj_size, int attn_o_proj_size,
                 int attn_max_qo_seq_len, int attn_max_kv_seq_len,
                 int attn_beam_size, const Tensor* attn_low_windows,
                 const Tensor* attn_high_windows, Tensor* o,
                 const Tensor* attn_qo_seqlen, const Tensor* attn_kv_seqlen,
                 Tensor* reserve_space) {
  int batch_size = q->dims()[0];
  int seq_len = q->dims()[1];
  auto dtype = platform::CudnnDataType<T>::type;
  auto comp_prec =
      dtype == CUDNN_DATA_DOUBLE ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;

  auto cudnn_handle = dev_ctx.cudnn_handle();

  size_t reserve_space_size = 0;
  void* reserve_space_ptr = nullptr;

  const std::string key = typeid(T).name();
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
  //   std::cout << attn_heads << ", " << attn_sm_scaler << ", " <<
  //   attn_vec_size
  //             << ", " << attn_q_proj_size << ", " << attn_o_proj_size << ", "
  //             << attn_max_qo_seq_len << ", " << attn_max_kv_seq_len << ", "
  //             << batch_size << ", " << attn_beam_size << std::endl;

  {
    // platform::RecordEvent record_event("cudnn_set_attn_descriptor",
    //                                    platform::EventRole::kInnerOp);
    // Setup Attention Desc
    // CUDNN_ATTN_QUERYMAP_ALL_TO_ONE
    // CUDNN_ATTN_ENABLE_PROJ_BIASES
    if (has_bias) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetAttnDescriptor(
          MHASingleton::Instance().Data(key).attn_desc,
          CUDNN_ATTN_ENABLE_PROJ_BIASES, attn_heads, attn_sm_scaler, dtype,
          comp_prec, CUDNN_DEFAULT_MATH, attn_dropout_desc, post_dropout_desc,
          attn_vec_size, attn_vec_size, attn_vec_size, attn_q_proj_size,
          attn_k_proj_size, attn_v_proj_size, attn_o_proj_size,
          attn_max_qo_seq_len, attn_max_kv_seq_len, batch_size,
          attn_beam_size));
    } else {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetAttnDescriptor(
          MHASingleton::Instance().Data(key).attn_desc,
          CUDNN_ATTN_QUERYMAP_ALL_TO_ONE, attn_heads, attn_sm_scaler, dtype,
          comp_prec, CUDNN_DEFAULT_MATH, attn_dropout_desc, post_dropout_desc,
          attn_vec_size, attn_vec_size, attn_vec_size, attn_q_proj_size,
          attn_k_proj_size, attn_v_proj_size, attn_o_proj_size,
          attn_max_qo_seq_len, attn_max_kv_seq_len, batch_size,
          attn_beam_size));
    }
  }

  {
    // platform::RecordEvent record_event("cudnn_get_multi_head_attn_buffers",
    //                                    platform::EventRole::kInnerOp);
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnGetMultiHeadAttnBuffers(
        cudnn_handle, MHASingleton::Instance().Data(key).attn_desc,
        &MHASingleton::Instance().Data(key).weights_size,
        &MHASingleton::Instance().Data(key).workspace_size,
        &reserve_space_size));
    // &MHASingleton::Instance().Data(key).reserve_size));
  }

  // TODO(rewang): ensure workspace size will not be increased
  if (MHASingleton::Instance().Data(key).workspace == nullptr) {
    MHASingleton::Instance().Data(key).workspace = memory::Alloc(
        dev_ctx, MHASingleton::Instance().Data(key).workspace_size);
  }

  // TODO(rewang): ensure reserve_size size will not be increased
  // if (MHASingleton::Instance().Data(key).reserve_space == nullptr) {
  //   MHASingleton::Instance().Data(key).reserve_space =
  //       memory::Alloc(dev_ctx,
  //       MHASingleton::Instance().Data(key).reserve_size);
  // }
  reserve_space_ptr = reserve_space->mutable_data(dev_ctx.GetPlace(), q->type(),
                                                  reserve_space_size);

  const int* attn_low_windows_data = attn_low_windows->data<int>();
  const int* attn_high_windows_data = attn_high_windows->data<int>();
  const int* attn_qo_seqlen_data = attn_qo_seqlen->data<int>();
  const int* attn_kv_seqlen_data = attn_kv_seqlen->data<int>();

  Tensor attn_low_windows_host;
  Tensor attn_high_windows_host;
  Tensor attn_qo_seqlen_host;
  Tensor attn_kv_seqlen_host;
  paddle::platform::CPUPlace cpu_place;
  if (!platform::is_cpu_place(attn_low_windows->place())) {
    attn_low_windows_host.mutable_data(cpu_place, attn_low_windows->type(),
                                       seq_len * sizeof(int));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(attn_low_windows_host.data<int>(),
                   reinterpret_cast<const void*>(attn_low_windows->data<int>()),
                   seq_len * sizeof(int), cudaMemcpyDeviceToHost));
    attn_low_windows_data = attn_low_windows_host.data<int>();
  }
  if (!platform::is_cpu_place(attn_high_windows->place())) {
    attn_high_windows_host.mutable_data(cpu_place, attn_high_windows->type(),
                                        seq_len * sizeof(int));

    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
        attn_high_windows_host.data<int>(),
        reinterpret_cast<const void*>(attn_high_windows->data<int>()),
        seq_len * sizeof(int), cudaMemcpyDeviceToHost));
    attn_high_windows_data = attn_high_windows_host.data<int>();
  }
  if (!platform::is_cpu_place(attn_qo_seqlen->place())) {
    attn_qo_seqlen_host.mutable_data(cpu_place, attn_qo_seqlen->type(),
                                     batch_size * sizeof(int));

    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(attn_qo_seqlen_host.data<int>(),
                   reinterpret_cast<const void*>(attn_qo_seqlen->data<int>()),
                   batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    attn_qo_seqlen_data = attn_qo_seqlen_host.data<int>();
  }
  if (!platform::is_cpu_place(attn_kv_seqlen->place())) {
    attn_kv_seqlen_host.mutable_data(cpu_place, attn_kv_seqlen->type(),
                                     batch_size * sizeof(int));

    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(attn_kv_seqlen_host.data<int>(),
                   reinterpret_cast<const void*>(attn_kv_seqlen->data<int>()),
                   batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    attn_kv_seqlen_data = attn_kv_seqlen_host.data<int>();
  }

#if 0
  std::vector<int> attn_qo_seqlen(q->dims()[0]);
  std::vector<int> attn_kv_seqlen(q->dims()[0]);
  std::vector<int> attn_low_windows(q->dims()[1]);
  std::vector<int> attn_high_windows(q->dims()[1]);
  // for (int i = 0; i < q->dims()[0]; i++) {
  //   attn_qo_seqlen[i] = q->dims()[1];
  //   attn_kv_seqlen[i] = q->dims()[1];
  // }
  // for (int i = 0; i < q->dims()[1]; i++) {
  //   attn_low_windows[i] = 0;
  //   attn_high_windows[i] = q->dims()[1];
  // }
  // TODO(rewang): use memory::Copy
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
      attn_qo_seqlen.data(),
      reinterpret_cast<const void*>(attn_qo_seqlen_input->data<int>()),
      q->dims()[0] * sizeof(int), cudaMemcpyDeviceToHost));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
      attn_kv_seqlen.data(),
      reinterpret_cast<const void*>(attn_kv_seqlen_input->data<int>()),
      q->dims()[0] * sizeof(int), cudaMemcpyDeviceToHost));

  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
      attn_low_windows.data(),
      reinterpret_cast<const void*>(attn_low_windows_input->data<int>()),
      q->dims()[1] * sizeof(int), cudaMemcpyDeviceToHost));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
      attn_high_windows.data(),
      reinterpret_cast<const void*>(attn_high_windows_input->data<int>()),
      q->dims()[1] * sizeof(int), cudaMemcpyDeviceToHost));
#endif

  //   PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
  //       kv_slen_host.data(), reinterpret_cast<const
  //       void*>(kv_slen->data<int>()), kv_slen->dims()[0] * sizeof(int),
  //       cudaMemcpyDeviceToHost));

  // const Tensor* attn_low_windows,
  // const Tensor* attn_high_windows, Tensor* o,
  // const Tensor* attn_qo_seqlen, const Tensor* attn_kv_seqlen,

  cudnnSeqDataAxis_t axes[CUDNN_SEQDATA_DIM_COUNT];  // [Batch, Beam, Seq, Vec]
  axes[0] = CUDNN_SEQDATA_BATCH_DIM;
  axes[1] = CUDNN_SEQDATA_TIME_DIM;
  axes[2] = CUDNN_SEQDATA_BEAM_DIM;
  axes[3] = CUDNN_SEQDATA_VECT_DIM;

  // q=[batch, seq_len, 1, embed_dim]

  int dimA[CUDNN_SEQDATA_DIM_COUNT];
  //   dimA[CUDNN_SEQDATA_VECT_DIM] = q->dims()[3];
  //   dimA[CUDNN_SEQDATA_BEAM_DIM] = q->dims()[2];
  //   dimA[CUDNN_SEQDATA_TIME_DIM] = q->dims()[1];
  //   dimA[CUDNN_SEQDATA_BATCH_DIM] = q->dims()[0];
  dimA[CUDNN_SEQDATA_VECT_DIM] = q->dims()[2];
  dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
  dimA[CUDNN_SEQDATA_TIME_DIM] = q->dims()[1];
  dimA[CUDNN_SEQDATA_BATCH_DIM] = q->dims()[0];
  {
    // platform::RecordEvent record_event("cudnn_set_seq_data_descriptor_q",
    //                                    platform::EventRole::kInnerOp);
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).q_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        attn_qo_seqlen_data, nullptr));
    // dimA, axes, batch_size * attn_beam_size, qo_slen_host.data(), nullptr));
  }

  //   dimA[CUDNN_SEQDATA_VECT_DIM] = k->dims()[3];
  //   dimA[CUDNN_SEQDATA_BEAM_DIM] = k->dims()[2];
  //   dimA[CUDNN_SEQDATA_TIME_DIM] = k->dims()[1];
  //   dimA[CUDNN_SEQDATA_BATCH_DIM] = k->dims()[0];
  dimA[CUDNN_SEQDATA_VECT_DIM] = k->dims()[2];
  dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
  dimA[CUDNN_SEQDATA_TIME_DIM] = k->dims()[1];
  dimA[CUDNN_SEQDATA_BATCH_DIM] = k->dims()[0];
  {
    // platform::RecordEvent record_event("cudnn_set_seq_data_descriptor_k",
    //                                    platform::EventRole::kInnerOp);
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).k_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        attn_kv_seqlen_data, nullptr));
    // dimA, axes, batch_size * attn_beam_size, kv_slen_host.data(), nullptr));
  }

  //   dimA[CUDNN_SEQDATA_VECT_DIM] = v->dims()[3];
  //   dimA[CUDNN_SEQDATA_BEAM_DIM] = v->dims()[2];
  //   dimA[CUDNN_SEQDATA_TIME_DIM] = v->dims()[1];
  //   dimA[CUDNN_SEQDATA_BATCH_DIM] = v->dims()[0];
  dimA[CUDNN_SEQDATA_VECT_DIM] = v->dims()[2];
  dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
  dimA[CUDNN_SEQDATA_TIME_DIM] = v->dims()[1];
  dimA[CUDNN_SEQDATA_BATCH_DIM] = v->dims()[0];
  {
    // platform::RecordEvent record_event("cudnn_set_seq_data_descriptor_v",
    //                                    platform::EventRole::kInnerOp);
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).v_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        attn_kv_seqlen_data, nullptr));
    // dimA, axes, batch_size * attn_beam_size, kv_slen_host.data(), nullptr));
  }

  //   dimA[CUDNN_SEQDATA_VECT_DIM] = o->dims()[3];
  //   dimA[CUDNN_SEQDATA_BEAM_DIM] = o->dims()[2];
  //   dimA[CUDNN_SEQDATA_TIME_DIM] = o->dims()[1];
  //   dimA[CUDNN_SEQDATA_BATCH_DIM] = o->dims()[0];
  dimA[CUDNN_SEQDATA_VECT_DIM] = o->dims()[2];
  dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
  dimA[CUDNN_SEQDATA_TIME_DIM] = o->dims()[1];
  dimA[CUDNN_SEQDATA_BATCH_DIM] = o->dims()[0];
  {
    // platform::RecordEvent record_event("cudnn_set_seq_data_descriptor_o",
    //                                    platform::EventRole::kInnerOp);
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSetSeqDataDescriptor(
        MHASingleton::Instance().Data(key).o_desc, dtype,
        CUDNN_SEQDATA_DIM_COUNT, dimA, axes, batch_size * attn_beam_size,
        attn_qo_seqlen_data, nullptr));
    // dimA, axes, batch_size * attn_beam_size, qo_slen_host.data(), nullptr));
  }

  //   std::vector<int> attn_low_windows =
  //       context.Attr<std::vector<int>>("attn_low_windows");
  //   std::vector<int> attn_high_windows =
  //       context.Attr<std::vector<int>>("attn_high_windows");

  // limin-todo: extract to op.cc
  // o->mutable_data<T>(context.GetPlace());
  const T* q_data = q->data<T>();
  const T* k_data = k->data<T>();
  const T* v_data = v->data<T>();
  const T* w_data = w->data<T>();

  T* o_data = o->data<T>();
  const T* residuals = nullptr;

  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnMultiHeadAttnForward(
      cudnn_handle, MHASingleton::Instance().Data(key).attn_desc, -1,
      attn_low_windows_data, attn_high_windows_data, qo_slen->data<int>(),
      kv_slen->data<int>(), MHASingleton::Instance().Data(key).q_desc, q_data,
      residuals, MHASingleton::Instance().Data(key).k_desc, k_data,
      MHASingleton::Instance().Data(key).v_desc, v_data,
      MHASingleton::Instance().Data(key).o_desc, o_data,
      MHASingleton::Instance().Data(key).weights_size, w_data,
      MHASingleton::Instance().Data(key).workspace_size,
      MHASingleton::Instance().Data(key).workspace->ptr(),
      // MHASingleton::Instance().Data(key).reserve_size,
      reserve_space_size, reserve_space_ptr));
  // MHASingleton::Instance().Data(key).reserve_space->ptr()));

  // PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnMultiHeadAttnForward(
  //     cudnn_handle, MHASingleton::Instance().Data(key).attn_desc, -1,
  //     attn_low_windows.data(), attn_high_windows.data(),
  //     qo_slen->data<int>(), qo_slen->data<int>(),
  //     MHASingleton::Instance().Data(key).q_desc, q_data, residuals,
  //     MHASingleton::Instance().Data(key).q_desc, k_data,
  //     MHASingleton::Instance().Data(key).q_desc, v_data,
  //     MHASingleton::Instance().Data(key).q_desc, o_data,
  //     MHASingleton::Instance().Data(key).weights_size, w_data,
  //     MHASingleton::Instance().Data(key).workspace_size,
  //     MHASingleton::Instance().Data(key).workspace->ptr(),
  //     // MHASingleton::Instance().Data(key).reserve_size,
  //     reserve_space_size, reserve_space_ptr));
  // // MHASingleton::Instance().Data(key).reserve_space->ptr()));
}

// template <typename DeviceContext, typename T>
// class MHAGradKernel : public framework::OpKernel<T> {
//  public:
//   void Compute(const framework::ExecutionContext& context) const override {

template <typename T>
void MHAGradKernel(const platform::CUDADeviceContext& dev_ctx,
                   const Tensor* dout, const Tensor* q, const Tensor* k,
                   const Tensor* v, const Tensor* w, const Tensor* qo_slen,
                   const Tensor* kv_slen, const Tensor* attn_low_windows,
                   const Tensor* attn_high_windows, Tensor* dq, Tensor* dk,
                   Tensor* dv, Tensor* dw, const Tensor* reserve_space) {
  //   auto& dev_ctx =
  //       context.template device_context<platform::CUDADeviceContext>();

  //   const Tensor* dout = context.Input<Tensor>(framework::GradVarName("O"));
  //   const Tensor* q = context.Input<Tensor>("Q");
  //   const Tensor* k = context.Input<Tensor>("K");
  //   const Tensor* v = context.Input<Tensor>("V");
  //   const Tensor* w = context.Input<Tensor>("W");
  //   const Tensor* qo_slen = context.Input<Tensor>("QO_Seqlen");
  //   const Tensor* kv_slen = context.Input<Tensor>("KV_Seqlen");

  //   Tensor* dq = context.Output<Tensor>(framework::GradVarName("Q"));
  //   Tensor* dk = context.Output<Tensor>(framework::GradVarName("K"));
  //   Tensor* dv = context.Output<Tensor>(framework::GradVarName("V"));
  //   Tensor* dw = context.Output<Tensor>(framework::GradVarName("W"));
  int batch_size = q->dims()[0];
  int seq_len = q->dims()[1];

  auto cudnn_handle = dev_ctx.cudnn_handle();

  auto reserve_space_size = reserve_space->memory_size();

  // TODO(Ming Huang): Need to come out a way to pass related variables from
  // FWD to BWD.
  const std::string key = typeid(T).name();

  //   std::vector<int> attn_low_windows =
  //       context.Attr<std::vector<int>>("attn_low_windows");
  //   std::vector<int> attn_high_windows =
  //       context.Attr<std::vector<int>>("attn_high_windows");

  const T* dout_data = dout->data<T>();
  const T* q_data = q->data<T>();
  const T* k_data = k->data<T>();
  const T* v_data = v->data<T>();
  const T* w_data = w->data<T>();

  const int* attn_low_windows_data = attn_low_windows->data<int>();
  const int* attn_high_windows_data = attn_high_windows->data<int>();

  Tensor attn_low_windows_host;
  Tensor attn_high_windows_host;
  paddle::platform::CPUPlace cpu_place;
  if (!platform::is_cpu_place(attn_low_windows->place())) {
    attn_low_windows_host.mutable_data(cpu_place, attn_low_windows->type(),
                                       seq_len * sizeof(int));
    attn_low_windows_data = attn_low_windows_host.data<int>();
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(attn_low_windows_host.data<int>(),
                   reinterpret_cast<const void*>(attn_low_windows->data<int>()),
                   seq_len * sizeof(int), cudaMemcpyDeviceToHost));
  }
  if (!platform::is_cpu_place(attn_high_windows->place())) {
    attn_high_windows_host.mutable_data(cpu_place, attn_high_windows->type(),
                                        seq_len * sizeof(int));
    attn_high_windows_data = attn_high_windows_host.data<int>();
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
        attn_high_windows_host.data<int>(),
        reinterpret_cast<const void*>(attn_high_windows->data<int>()),
        seq_len * sizeof(int), cudaMemcpyDeviceToHost));
  }

  // limin-todo:
  //   dq->mutable_data<T>(context.GetPlace());
  //   dk->mutable_data<T>(context.GetPlace());
  //   dv->mutable_data<T>(context.GetPlace());
  //   dw->mutable_data<T>(context.GetPlace());

  // todo: e.g., when dq is null, xxx

  T* dq_data = dq->data<T>();
  T* dk_data = dk->data<T>();
  T* dv_data = dv->data<T>();
  T* dw_data = dw->data<T>();

#if 0
  std::vector<int> attn_low_windows(q->dims()[1]);
  std::vector<int> attn_high_windows(q->dims()[1]);
  // for (int i = 0; i < q->dims()[1]; i++) {
  //   attn_low_windows[i] = 0;
  //   attn_high_windows[i] = q->dims()[1];
  // }
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
      attn_low_windows_data(),
      reinterpret_cast<const void*>(attn_low_windows_input->data<int>()),
      q->dims()[1] * sizeof(int), cudaMemcpyDeviceToHost));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemcpy(
      attn_high_windows.data(),
      reinterpret_cast<const void*>(attn_high_windows_input->data<int>()),
      q->dims()[1] * sizeof(int), cudaMemcpyDeviceToHost));
#endif

  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnMultiHeadAttnBackwardData(
      cudnn_handle, MHASingleton::Instance().Data(key).attn_desc,
      attn_low_windows_data, attn_high_windows_data, qo_slen->data<int>(),
      kv_slen->data<int>(), MHASingleton::Instance().Data(key).o_desc,
      dout_data, MHASingleton::Instance().Data(key).q_desc, dq_data, q_data,
      MHASingleton::Instance().Data(key).k_desc, dk_data, k_data,
      MHASingleton::Instance().Data(key).v_desc, dv_data, v_data,
      MHASingleton::Instance().Data(key).weights_size, w_data,
      MHASingleton::Instance().Data(key).workspace_size,
      MHASingleton::Instance().Data(key).workspace->ptr(), reserve_space_size,
      const_cast<T*>(reserve_space->template data<T>())));
  // MHASingleton::Instance().Data(key).reserve_size,
  // MHASingleton::Instance().Data(key).reserve_space->ptr()));

  // PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnMultiHeadAttnBackwardData(
  //     cudnn_handle, MHASingleton::Instance().Data(key).attn_desc,
  //     attn_low_windows.data(), attn_high_windows.data(),
  //     qo_slen->data<int>(), qo_slen->data<int>(),
  //     MHASingleton::Instance().Data(key).q_desc, dout_data,
  //     MHASingleton::Instance().Data(key).q_desc, dq_data, q_data,
  //     MHASingleton::Instance().Data(key).q_desc, dk_data, k_data,
  //     MHASingleton::Instance().Data(key).q_desc, dv_data, v_data,
  //     MHASingleton::Instance().Data(key).weights_size, w_data,
  //     MHASingleton::Instance().Data(key).workspace_size,
  //     MHASingleton::Instance().Data(key).workspace->ptr(),
  //     reserve_space_size, const_cast<T*>(reserve_space->template
  //     data<T>())));
  // // MHASingleton::Instance().Data(key).reserve_size,
  // // MHASingleton::Instance().Data(key).reserve_space->ptr()));

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
          reserve_space_size,
          const_cast<T*>(reserve_space->template data<T>())));
  // MHASingleton::Instance().Data(key).reserve_size,
  // MHASingleton::Instance().Data(key).reserve_space->ptr()));

  // PADDLE_ENFORCE_CUDA_SUCCESS(
  //     platform::dynload::cudnnMultiHeadAttnBackwardWeights(
  //         cudnn_handle, MHASingleton::Instance().Data(key).attn_desc,
  //         CUDNN_WGRAD_MODE_SET, MHASingleton::Instance().Data(key).q_desc,
  //         q_data, MHASingleton::Instance().Data(key).q_desc, k_data,
  //         MHASingleton::Instance().Data(key).q_desc, v_data,
  //         MHASingleton::Instance().Data(key).q_desc, dout_data,
  //         MHASingleton::Instance().Data(key).weights_size, w_data, dw_data,
  //         MHASingleton::Instance().Data(key).workspace_size,
  //         MHASingleton::Instance().Data(key).workspace->ptr(),
  //         reserve_space_size,
  //         const_cast<T*>(reserve_space->template data<T>())));
  // // MHASingleton::Instance().Data(key).reserve_size,
  // // MHASingleton::Instance().Data(key).reserve_space->ptr()))
}
#endif

}  // namespace operators
}  // namespace paddle
