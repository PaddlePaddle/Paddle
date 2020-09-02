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
template <typename T>
class EmbEltwiseLayernormPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit EmbEltwiseLayernormPluginDynamic(std::vector<float*> input_embs,
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

  EmbEltwiseLayernormPluginDynamic(void const* serial_data,
                                   size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &emb_sizes_);

    embs_gpu_.resize(emb_sizes_.size());
    embs_.resize(emb_sizes_.size());
    for (size_t i = 0; i < emb_sizes_.size(); i++) {
      cudaMalloc(&embs_gpu_[i], sizeof(float) * emb_sizes_[i]);
      cudaMemcpy(embs_gpu_[i], serial_data, emb_sizes_[i] * sizeof(float),
                 cudaMemcpyHostToDevice);
      reinterpret_cast<char const*&>(serial_data) +=
          emb_sizes_[i] * sizeof(float);
      serial_length -= emb_sizes_[i] * sizeof(float);
      embs_[i] = nullptr;
    }
    DeserializeValue(&serial_data, &serial_length, &bias_size_);
    DeserializeValue(&serial_data, &serial_length, &scale_size_);

    cudaMalloc(&bias_gpu_, sizeof(float) * bias_size_);
    cudaMemcpy(bias_gpu_, serial_data, bias_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    bias_ = nullptr;
    reinterpret_cast<char const*&>(serial_data) += bias_size_ * sizeof(float);
    serial_length -= bias_size_ * sizeof(float);

    cudaMalloc(&scale_gpu_, sizeof(float) * scale_size_);
    cudaMemcpy(scale_gpu_, serial_data, scale_size_ * sizeof(float),
               cudaMemcpyHostToDevice);
    scale_ = nullptr;
    reinterpret_cast<char const*&>(serial_data) += scale_size_ * sizeof(float);
    serial_length -= scale_size_ * sizeof(float);

    DeserializeValue(&serial_data, &serial_length, &hidden_size_);
    DeserializeValue(&serial_data, &serial_length, &eps_);
  }

  nvinfer1::IPluginV2DynamicExt* clone() const override {
    auto ptr = new EmbEltwiseLayernormPluginDynamic(
        embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_, hidden_size_,
        eps_);
    ptr->embs_gpu_ = embs_gpu_;
    ptr->bias_gpu_ = bias_gpu_;
    ptr->scale_gpu_ = scale_gpu_;
    return ptr;
  }

  const char* getPluginType() const override {
    return "fused_embedding_eltwise_layernorm_plugin";
  }
  int getNbOutputs() const override { return 1; }
  int initialize() override;

  size_t getSerializationSize() const override {
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
    // sum_num += SerializedSize(with_fp16_);

    return sum_num;
  }

  void terminate() override;
  void serialize(void* buffer) const override {
    // SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, emb_sizes_);
    for (size_t i = 0; i < emb_sizes_.size(); i++) {
      SerializeCudaPointer(&buffer, embs_gpu_[i], emb_sizes_[i]);
    }
    SerializeValue(&buffer, bias_size_);
    SerializeValue(&buffer, scale_size_);
    SerializeCudaPointer(&buffer, bias_gpu_, bias_size_);
    SerializeCudaPointer(&buffer, scale_gpu_, scale_size_);
    SerializeValue(&buffer, hidden_size_);
    SerializeValue(&buffer, eps_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* in_out,
                                 int nb_inputs, int nb_outputs) override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nb_outputs) override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
              const nvinfer1::PluginTensorDesc* output_desc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const override;

  void destroy() override { delete this; }

 private:
  std::vector<float*> embs_;
  float* bias_;
  float* scale_;

  // data on devices
  float* bias_gpu_;
  float* scale_gpu_;
  std::vector<float*> embs_gpu_;

  std::vector<int> emb_sizes_;
  int bias_size_;
  int scale_size_;
  int hidden_size_;
  float eps_;
};

class EmbEltwiseLayernormPluginV2Creator : public nvinfer1::IPluginCreator {
 public:
  EmbEltwiseLayernormPluginV2Creator() {}
  const char* getPluginName() const override {
    return "fused_embedding_eltwise_layernorm_plugin";
  }

  const char* getPluginVersion() const override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length) override {
    return new EmbEltwiseLayernormPluginDynamic<float>(serial_data,
                                                       serial_length);
  }

  void setPluginNamespace(const char* lib_namespace) override {
    plugin_namespace_ = lib_namespace;
  }

  const char* getPluginNamespace() const override {
    return plugin_namespace_.c_str();
  }

 private:
  std::string plugin_namespace_;
  std::string plugin_name_;
  nvinfer1::PluginFieldCollection field_collection_;
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};

REGISTER_TRT_PLUGIN_V2(EmbEltwiseLayernormPluginV2Creator);

#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
