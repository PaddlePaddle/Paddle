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

  EmbEltwiseLayernormPluginDynamic(void const* serialData,
                                   size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &embs_);
    DeserializeValue(&serialData, &serialLength, &bias_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &emb_sizes_);
    DeserializeValue(&serialData, &serialLength, &bias_size_);
    DeserializeValue(&serialData, &serialLength, &scale_size_);
    DeserializeValue(&serialData, &serialLength, &hidden_size_);
    DeserializeValue(&serialData, &serialLength, &eps_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new EmbEltwiseLayernormPluginDynamic(
        embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_, hidden_size_,
        eps_);
  }

  const char* getPluginType() const override {
    return "fused_embedding_eltwise_layernorm_plugin";
  }
  int getNbOutputs() const override { return 1; }
  int initialize() override;

  size_t getSerializationSize() const override {
    return sizeof(float*) * embs_.size() + sizeof(float*) * 2 +
           sizeof(int) * emb_sizes_.size() + sizeof(int) * 3 + sizeof(float);
  }
  void serialize(void* buffer) const override {
    SerializeValue(&buffer, embs_);
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, emb_sizes_);
    SerializeValue(&buffer, bias_size_);
    SerializeValue(&buffer, scale_size_);
    SerializeValue(&buffer, hidden_size_);
    SerializeValue(&buffer, eps_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

  void destroy() override { delete this; }

 private:
  std::vector<float*> embs_;
  float* bias_;
  float* scale_;

  // data on devices
  float* bias_gpu_;
  float* scale_gpu_;
  std::vector<T*> embs_gpu_;

  std::vector<int> emb_sizes_;
  int bias_size_;
  int scale_size_;
  int hidden_size_;
  float eps_;
};

class EmbEltwiseLayernormPluginV2Creator : public nvinfer1::IPluginCreator {
 public:
  EmbEltwiseLayernormPluginV2Creator() {
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "embs_", nullptr, nvinfer1::PluginFieldType::kUNKNOWN, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "bias_", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "scale_", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "emb_sizes_", nullptr, nvinfer1::PluginFieldType::kUNKNOWN, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "bias_size_", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "scale_size_", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "hidden_size_", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "eps_", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
  }
  const char* getPluginName() const override {
    return "fused_embedding_eltwise_layernorm_pluginpaddle_trt";
  }

  const char* getPluginVersion() const override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &mFieldCollection;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    std::vector<float*> embs_;
    float* bias_;
    float* scale_;
    std::vector<int> emb_sizes_;
    int bias_size_, scale_size_, hidden_size_;
    float eps_;
    const nvinfer1::PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
      const char* attrName = fields[i].name;
      if (!strcmp(attrName, "embs_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kUNKNOWN);
        embs_ = *(static_cast<const std::vector<float*>*>(fields[i].data));
      }
      if (!strcmp(attrName, "bias_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
        const float* temp_bias_ = (static_cast<const float*>(fields[i].data));
        bias_ = const_cast<float*>(temp_bias_);
      }
      if (!strcmp(attrName, "scale_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
        const float* temp_scale_ = (static_cast<const float*>(fields[i].data));
        scale_ = const_cast<float*>(temp_scale_);
      }
      if (!strcmp(attrName, "emb_sizes_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kUNKNOWN);
        emb_sizes_ = *(static_cast<const std::vector<int>*>(fields[i].data));
      }
      if (!strcmp(attrName, "bias_size_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
        bias_size_ = *(static_cast<const int*>(fields[i].data));
      }
      if (!strcmp(attrName, "scale_size_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
        scale_size_ = *(static_cast<const int*>(fields[i].data));
      }
      if (!strcmp(attrName, "hidden_size_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
        hidden_size_ = *(static_cast<const int*>(fields[i].data));
      }
      if (!strcmp(attrName, "eps_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
        eps_ = *(static_cast<const float*>(fields[i].data));
      }
    }

    return new EmbEltwiseLayernormPluginDynamic<float>(
        embs_, bias_, scale_, emb_sizes_, bias_size_, scale_size_, hidden_size_,
        eps_);
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) override {
    auto plugin =
        new EmbEltwiseLayernormPluginDynamic<float>(serialData, serialLength);
    return plugin;
  }

  void setPluginNamespace(const char* libNamespace) override {
    mNamespace = libNamespace;
  }

  const char* getPluginNamespace() const override { return mNamespace.c_str(); }

 private:
  std::string mNamespace{"paddle_trt"};
  std::string mPluginName;
  nvinfer1::PluginFieldCollection mFieldCollection;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(EmbEltwiseLayernormPluginV2Creator);

#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
