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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  //
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

class EmbEltwiseLayernormPluginDynamicImplBase {
 public:
  EmbEltwiseLayernormPluginDynamicImplBase() {}
  virtual ~EmbEltwiseLayernormPluginDynamicImplBase() {}

  virtual int initialize() = 0;
  virtual void terminate() = 0;
  virtual int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                      const nvinfer1::PluginTensorDesc* outputDesc,
                      const void* const* inputs, void* const* outputs,
                      void* workspace, cudaStream_t stream) = 0;
  virtual void shareGPUData(
      const EmbEltwiseLayernormPluginDynamicImplBase* anthor) = 0;
};

template <typename T>
class EmbEltwiseLayernormPluginDynamicImpl
    : public EmbEltwiseLayernormPluginDynamicImplBase {
 public:
  explicit EmbEltwiseLayernormPluginDynamicImpl(std::vector<float*> input_embs,
                                                float* bias, float* scale,
                                                std::vector<int> emb_sizes,
                                                int bias_size, int scale_size,
                                                int hidden_size, float eps)
      : embs_(input_embs),
        bias_(bias),
        scale_(scale),
        emb_sizes_(emb_sizes),
        bias_size_(bias_size),
        scale_size_(scale_size),
        hidden_size_(hidden_size),
        eps_(eps) {}

  ~EmbEltwiseLayernormPluginDynamicImpl();

  int initialize();
  void terminate();
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT;
  void shareGPUData(const EmbEltwiseLayernormPluginDynamicImplBase* anthor);

 private:
  std::vector<float*> embs_;
  float* bias_{nullptr};
  float* scale_{nullptr};

  // data on devices
  float* bias_gpu_{nullptr};
  float* scale_gpu_{nullptr};
  std::vector<T*> embs_gpu_;

  std::vector<int> emb_sizes_;
  int bias_size_;
  int scale_size_;
  int hidden_size_;
  float eps_;

  framework::Tensor in_ptr_tensor_, emb_ptr_tensor_;
  int device_id_{0};
  bool is_initialized_{false};
};

class EmbEltwiseLayernormPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit EmbEltwiseLayernormPluginDynamic(std::vector<float*> input_embs,
                                            float* bias, float* scale,
                                            std::vector<int> emb_sizes,
                                            int bias_size, int scale_size,
                                            int hidden_size, float eps,
                                            bool with_fp16)
      : embs_(input_embs),
        bias_(bias),
        scale_(scale),
        emb_sizes_(emb_sizes),
        bias_size_(bias_size),
        scale_size_(scale_size),
        hidden_size_(hidden_size),
        eps_(eps),
        own_host_buff_(false) {
    with_fp16_ = with_fp16;
    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
      VLOG(1) << "TRT Plugin DataType selected. EmbEltwiseLayerNorm-->fp16";
      impl_ = new EmbEltwiseLayernormPluginDynamicImpl<half>(
          embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_,
          hidden_size_, eps_);
#else
      PADDLE_THROW(platform::errors::Fatal(
          "The Ernie(Bert) tensorRT plugin should be "
          "complied with CUDA version >= 10.0 when running with fp16. "
          "Please recomplie it or try to use fp32 by set "
          "config.EnableTensorRtEngine(1 << 30, 1, 5, "
          "AnalysisConfig::Precision::kFloat32, false, false) "));
#endif
    } else {
      VLOG(1) << "TRT Plugin DataType selected. EmbEltwiseLayerNorm-->fp32";
      impl_ = new EmbEltwiseLayernormPluginDynamicImpl<float>(
          embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_,
          hidden_size_, eps_);
    }
  }

  EmbEltwiseLayernormPluginDynamic(void const* serial_data,
                                   size_t serial_length)
      : own_host_buff_(true) {
    DeserializeValue(&serial_data, &serial_length, &emb_sizes_);

    embs_.resize(emb_sizes_.size());
    for (size_t i = 0; i < emb_sizes_.size(); i++) {
      auto size = emb_sizes_[i];
      auto ptr = new float[size];
      memcpy(ptr, serial_data, sizeof(float) * size);
      embs_[i] = ptr;
      reinterpret_cast<char const*&>(serial_data) +=
          emb_sizes_[i] * sizeof(float);
      serial_length -= emb_sizes_[i] * sizeof(float);
    }
    DeserializeValue(&serial_data, &serial_length, &bias_size_);
    DeserializeValue(&serial_data, &serial_length, &scale_size_);

    if (bias_size_) {
      bias_ = new float[bias_size_];
      memcpy(bias_, serial_data, sizeof(float) * bias_size_);
    }
    reinterpret_cast<char const*&>(serial_data) += bias_size_ * sizeof(float);
    serial_length -= bias_size_ * sizeof(float);

    if (scale_size_) {
      scale_ = new float[scale_size_];
      memcpy(scale_, serial_data, sizeof(float) * scale_size_);
    }
    reinterpret_cast<char const*&>(serial_data) += scale_size_ * sizeof(float);
    serial_length -= scale_size_ * sizeof(float);

    DeserializeValue(&serial_data, &serial_length, &hidden_size_);
    DeserializeValue(&serial_data, &serial_length, &eps_);
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);

    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
      impl_ = new EmbEltwiseLayernormPluginDynamicImpl<half>(
          embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_,
          hidden_size_, eps_);
#else
      PADDLE_THROW(platform::errors::Fatal(
          "The Ernie(Bert) tensorRT plugin should be "
          "complied with CUDA version >= 10.0 when running with fp16. "
          "Please recomplie it or try to use fp32 by set "
          "config.EnableTensorRtEngine(1 << 30, 1, 5, "
          "AnalysisConfig::Precision::kFloat32, false, false) "));
#endif
    } else {
      impl_ = new EmbEltwiseLayernormPluginDynamicImpl<float>(
          embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_,
          hidden_size_, eps_);
    }
  }

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    auto ptr = new EmbEltwiseLayernormPluginDynamic(
        embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_, hidden_size_,
        eps_, with_fp16_);
    ptr->shareGPUData(this);
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "fused_embedding_eltwise_layernorm_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;
  void terminate() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    int sum_num = 0;
    sum_num += SerializedSize(emb_sizes_);

    for (size_t i = 0; i < emb_sizes_.size(); i++) {
      sum_num += emb_sizes_[i] * sizeof(float);
    }

    sum_num += SerializedSize(bias_size_);
    sum_num += SerializedSize(scale_size_);

    sum_num += (bias_size_ + scale_size_) * sizeof(float);
    sum_num += SerializedSize(hidden_size_);
    sum_num += SerializedSize(eps_);
    sum_num += SerializedSize(with_fp16_);

    return sum_num;
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, emb_sizes_);
    for (size_t i = 0; i < emb_sizes_.size(); i++) {
      auto size = emb_sizes_[i];
      for (int j = 0; j < size; ++j) {
        SerializeValue(&buffer, embs_[i][j]);
      }
    }
    SerializeValue(&buffer, bias_size_);
    SerializeValue(&buffer, scale_size_);
    for (int i = 0; i < bias_size_; ++i) {
      SerializeValue(&buffer, bias_[i]);
    }

    for (int i = 0; i < scale_size_; ++i) {
      SerializeValue(&buffer, scale_[i]);
    }

    SerializeValue(&buffer, hidden_size_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, with_fp16_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs,
                                 int nb_outputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nb_outputs) TRT_NOEXCEPT override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc* output_desc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* input_types,
      int nb_inputs) const TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override {
    if (own_host_buff_) {
      for (auto ptr : embs_) {
        delete[] ptr;
      }
      delete[] bias_;
      delete[] scale_;
    }

    delete impl_;
    delete this;
  }

 private:
  std::vector<float*> embs_;
  float* bias_;
  float* scale_;

  std::vector<int> emb_sizes_;
  int bias_size_;
  int scale_size_;
  int hidden_size_;
  float eps_;

  bool own_host_buff_{false};
  EmbEltwiseLayernormPluginDynamicImplBase* impl_{nullptr};

  void shareGPUData(const EmbEltwiseLayernormPluginDynamic* anthor) {
    impl_->shareGPUData(anthor->impl_);
  }
};

class EmbEltwiseLayernormPluginDynamicCreator
    : public nvinfer1::IPluginCreator {
 public:
  EmbEltwiseLayernormPluginDynamicCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "fused_embedding_eltwise_layernorm_plugin";
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

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    return new EmbEltwiseLayernormPluginDynamic(serial_data, serial_length);
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

REGISTER_TRT_PLUGIN_V2(EmbEltwiseLayernormPluginDynamicCreator);

#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
