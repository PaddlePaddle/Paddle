// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)
class QkvToContextPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit QkvToContextPluginDynamic(
      int hidden, int head_number, int head_size, float scale, bool with_fp16)
      : hidden_(hidden),
        head_number_(head_number),
        head_size_(head_size),
        scale_(scale) {
    with_fp16_ = with_fp16;
  }

  QkvToContextPluginDynamic(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &hidden_);
    DeserializeValue(&serial_data, &serial_length, &head_number_);
    DeserializeValue(&serial_data, &serial_length, &head_size_);
    DeserializeValue(&serial_data, &serial_length, &scale_);
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
    DeserializeValue(&serial_data, &serial_length, &cublas_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    // printf("@@ clone begin \r\n");
    QkvToContextPluginDynamic* ptr = new QkvToContextPluginDynamic(
        hidden_, head_number_, head_size_, scale_, with_fp16_);
    ptr->cublas_ = cublas_;
    ptr->operation_desc_qk_=operation_desc_qk_;
    ptr->q_desc_ = q_desc_;
    ptr->k_desc_=k_desc_;
    ptr->v_desc_=v_desc_;
    ptr->qk_desc_=qk_desc_;
    ptr->qk_bias_desc_=qk_bias_desc_;
    ptr->qkv_desc_=qkv_desc_;
    ptr->algo_=algo_;
    ptr->algo_qkv_=algo_qkv_;
    // printf("@@ clone end");
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "qkv_to_context_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(hidden_) + SerializedSize(head_number_) +
           SerializedSize(head_size_) + SerializedSize(scale_) +
           SerializedSize(with_fp16_) + SerializedSize(cublas_);
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, hidden_);
    SerializeValue(&buffer, head_number_);
    SerializeValue(&buffer, head_size_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, cublas_);
  }

  nvinfer1::DimsExprs getOutputDimensions(int output_index,
                                          const nvinfer1::DimsExprs* inputs,
                                          int nb_inputs,
                                          nvinfer1::IExprBuilder& expr_builder)
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nb_outputs) TRT_NOEXCEPT override {
  //TODO wangbojun
  //choose gemm value;

  //configure qk gemm
    if(with_fp16_){
      // int device_id;
      // cudaGetDevice(&device_id);
      // auto *device_ctx = static_cast<phi::GPUContext *>(
      //   platform::DeviceContextPool::Instance().Get(
      //     platform::CUDAPlace(device_id)));
      // const phi::GPUContext &dev_ctx = *device_ctx;
      platform::dynload::cublasLtCreate(&cublas_);
      int seq_len = in[0].desc.dims.d[1];
      const int padding_num=8;
      seq_len = (seq_len + padding_num - 1) / padding_num * padding_num; //padding
      int batchNum = in[0].desc.dims.d[0] * head_number_;
      // printf("@@@@ in config seq_len %d, batch %d. \r\n",seq_len,batchNum);
      int64_t strideq=seq_len*head_size_;
      int64_t stridek=seq_len*head_size_;
      int64_t strideqk=seq_len*seq_len;
      int64_t stridev=seq_len*head_size_;
      int64_t strideqkv=seq_len*head_size_;

      // printf("@@@ in config, head_size_ %d \r\n ", head_size_);
      bool q_trans = false;
      bool k_trans = true;
      bool qk_trans = false;
      bool v_trans = false;
      cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;
      cublasOperation_t transq = q_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
      cublasOperation_t transk = k_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
      cublasOperation_t transqk = qk_trans ? CUBLAS_OP_T : CUBLAS_OP_N;
      cublasOperation_t transv = v_trans ? CUBLAS_OP_T : CUBLAS_OP_N;

      int64_t q_M = q_trans ? head_size_ : seq_len;
      int64_t q_K = q_trans ? seq_len : head_size_;

      int64_t k_K = k_trans ? seq_len : head_size_;
      int64_t k_N = k_trans ? head_size_ : seq_len;

      int64_t v_M = q_trans ? head_size_ : seq_len;
      int64_t v_K = q_trans ? seq_len : head_size_;


      cudaDataType_t q_type, k_type, qk_type, scale_type, qk_bias_type, v_type, qkv_type;
      cublasComputeType_t compute_type;
      compute_type=CUBLAS_COMPUTE_16F;
      q_type=CUDA_R_16F;
      k_type=CUDA_R_16F;
      v_type=CUDA_R_16F;
      qk_type=CUDA_R_16F;
      qk_bias_type=CUDA_R_16F;
      qkv_type=CUDA_R_16F;
      scale_type=CUDA_R_16F;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
        &q_desc_, q_type, q_M, q_K, head_size_));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
        q_desc_,
        CUBLASLT_MATRIX_LAYOUT_TYPE,
        &q_type,
        sizeof(q_type)));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatrixLayoutSetAttribute(
            q_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          q_desc_, 
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
          &(batchNum), 
          sizeof(batchNum)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          q_desc_,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
          &(strideq),
          sizeof(strideq)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
        &k_desc_, k_type, k_K, k_N, head_size_));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatrixLayoutSetAttribute(
            k_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          k_desc_,
          CUBLASLT_MATRIX_LAYOUT_TYPE,
          &k_type,
          sizeof(k_type)
      ));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          k_desc_, 
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
          &(batchNum), 
          sizeof(batchNum)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          k_desc_,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
          &(stridek),
          sizeof(stridek)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
        &v_desc_, v_type, v_M, v_K, head_size_));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
        v_desc_,
        CUBLASLT_MATRIX_LAYOUT_TYPE,
        &v_type,
        sizeof(v_type)));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatrixLayoutSetAttribute(
            v_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          v_desc_, 
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
          &(batchNum), 
          sizeof(batchNum)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          v_desc_,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
          &(stridev),
          sizeof(stridev)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &qk_desc_, 
          qk_type, 
          seq_len, seq_len, seq_len));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatrixLayoutSetAttribute(
            qk_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          qk_desc_,
          CUBLASLT_MATRIX_LAYOUT_TYPE,
          &qk_type,
          sizeof(qk_type)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          qk_desc_, 
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
          &(batchNum), 
          sizeof(batchNum)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          qk_desc_,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
          &(strideqk),
          sizeof(strideqk)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
          &qk_bias_desc_,
          qk_bias_type,
          seq_len, seq_len, seq_len));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          qk_bias_desc_,
          CUBLASLT_MATRIX_LAYOUT_TYPE,
          &qk_bias_type,
          sizeof(qk_bias_type)));
      PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatrixLayoutSetAttribute(
        qk_bias_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          qk_bias_desc_, 
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
          &(batchNum), 
          sizeof(batchNum)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          qk_bias_desc_,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
          &(strideqk),
          sizeof(strideqk)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutCreate(
        &qkv_desc_, qkv_type, v_M, v_K, head_size_));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
        qkv_desc_,
        CUBLASLT_MATRIX_LAYOUT_TYPE,
        &qkv_type,
        sizeof(qkv_type)));
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cublasLtMatrixLayoutSetAttribute(
            qkv_desc_, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof( rowOrder ) ) );
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          qkv_desc_, 
          CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, 
          &(batchNum), 
          sizeof(batchNum)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatrixLayoutSetAttribute(
          qkv_desc_,
          CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
          &(strideqkv),
          sizeof(strideqkv)));

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescCreate(
        &operation_desc_qk_, compute_type, scale_type));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescSetAttribute(
        operation_desc_qk_,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transq, 
        sizeof(transq)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescSetAttribute(
        operation_desc_qk_,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transk,
        sizeof(transk)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescCreate(
        &operation_desc_qkv_, compute_type, scale_type));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescSetAttribute(
        operation_desc_qkv_,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transqk, 
        sizeof(transqk)));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cublasLtMatmulDescSetAttribute(
        operation_desc_qkv_,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &transv,
        sizeof(transv)));

      cublasLtMatmulPreference_t preference;
      PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cublasLtMatmulPreferenceCreate(&preference));
      // int qk_workspace_size=4*1024*1024;
      // PADDLE_ENFORCE_GPU_SUCCESS(
      //   platform::dynload::cublasLtMatmulPreferenceSetAttribute(
      //     preference,
      //     CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      //     &qk_workspace_size,
      //     sizeof(qk_workspace_size)));
      int returned_results = 0;
      const int requested_algo_count = 10;
      std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results(
          requested_algo_count);
      PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulAlgoGetHeuristic(
          cublas_,
          operation_desc_qk_,
          q_desc_,
          k_desc_,
          qk_bias_desc_,
          qk_desc_,
          preference,
          requested_algo_count,
          heuristic_results.data(),
          &returned_results));
      PADDLE_ENFORCE_GT(
          returned_results,
          0,
          platform::errors::Unavailable("No GEMM epilogue algorithm support!"));
      algo_ = heuristic_results[0].algo;
      cublasLtMatmulPreference_t preference_qkv;
      PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cublasLtMatmulPreferenceCreate(&preference_qkv));
      // int qk_workspace_size=4*1024*1024;
      // PADDLE_ENFORCE_GPU_SUCCESS(
      //   platform::dynload::cublasLtMatmulPreferenceSetAttribute(
      //     preference,
      //     CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      //     &qk_workspace_size,
      //     sizeof(qk_workspace_size)));
      int returned_results_qkv = 0;
      const int requested_algo_count_qkv = 10;
      std::vector<cublasLtMatmulHeuristicResult_t> heuristic_results_qkv(
          requested_algo_count_qkv);
      PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::cublasLtMatmulAlgoGetHeuristic(
          cublas_,
          operation_desc_qkv_,
          qk_desc_,
          v_desc_,
          qkv_desc_,
          qkv_desc_,
          preference_qkv,
          requested_algo_count_qkv,
          heuristic_results_qkv.data(),
          &returned_results_qkv));
      PADDLE_ENFORCE_GT(
          returned_results_qkv,
          0,
          platform::errors::Unavailable("No GEMM epilogue algorithm support!"));
      algo_qkv_ = heuristic_results_qkv[0].algo;

      platform::dynload::cublasLtDestroy(cublas_);
    //TODO speed test. wangbojun
    }
  }

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const TRT_NOEXCEPT override {
    auto input_dims = inputs[0].dims;
    const int batch = input_dims.d[0]; // batch = for swin, batch in input = image_batch * window_number
    // printf("@@@ in get worksapce, batch :%d \r\n", batch);
    int seq_len = input_dims.d[1];
    const int padding_num=8;
    seq_len = (seq_len + padding_num - 1) / padding_num * padding_num;
    const int input_num = batch * seq_len * 3 * head_number_ * head_size_;
    const size_t qk_temp_ptr_size = batch * head_number_ * seq_len * seq_len + input_num;
    const size_t biasqk_size =  batch * head_number_* seq_len* seq_len;
    const size_t cublaslt_workspace_size=4*1024*1024; // workspace for cublaslt, 4M for now
    if(with_fp16_){
      return sizeof(half)*(qk_temp_ptr_size+biasqk_size+2*cublaslt_workspace_size);
    } else {
      return sizeof(float)*(qk_temp_ptr_size+biasqk_size+2*cublaslt_workspace_size);
    }
    // return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

  void attachToContext(cudnnContext* cudnnContext,
                      cublasContext* cublasContext,
                      nvinfer1::IGpuAllocator* gpuAllocator)
    TRT_NOEXCEPT override;

  void detachFromContext() TRT_NOEXCEPT override;

 private:
  int hidden_;
  int head_number_;
  int head_size_;
  float scale_;
  cublasLtHandle_t cublas_{nullptr};
  cublasLtMatmulDesc_t operation_desc_qk_ = NULL;
  cublasLtMatmulDesc_t operation_desc_qkv_ = NULL;
  cublasLtMatrixLayout_t q_desc_ = NULL;
  cublasLtMatrixLayout_t k_desc_ = NULL;
  cublasLtMatrixLayout_t v_desc_ = NULL;
  cublasLtMatrixLayout_t qk_desc_ = NULL;
  cublasLtMatrixLayout_t qk_bias_desc_ = NULL;
  cublasLtMatrixLayout_t qkv_desc_ = NULL;
  cublasLtMatmulAlgo_t algo_;
  cublasLtMatmulAlgo_t algo_qkv_;

};

class QkvToContextPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  QkvToContextPluginDynamicCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "qkv_to_context_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc)
      TRT_NOEXCEPT override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    auto plugin = new QkvToContextPluginDynamic(serial_data, serial_length);
    return plugin;
  }

  void setPluginNamespace(const char* lib_namespace) TRT_NOEXCEPT override {
    plugin_namespace_ = lib_namespace;
  }

  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return plugin_namespace_.c_str();
  }

 private:
  std::string plugin_namespace_;
  std::string plugin_name_;
  nvinfer1::PluginFieldCollection field_collection_;
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};
REGISTER_TRT_PLUGIN_V2(QkvToContextPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
