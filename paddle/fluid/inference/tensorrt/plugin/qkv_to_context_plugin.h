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
  explicit QkvToContextPluginDynamic(int hidden, int head_number, int head_size,
                                     float scale, bool ban_fp16)
      : hidden_(hidden),
        head_number_(head_number),
        head_size_(head_size),
        scale_(scale),
        ban_fp16_(ban_fp16) {}

  QkvToContextPluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &hidden_);
    DeserializeValue(&serialData, &serialLength, &head_number_);
    DeserializeValue(&serialData, &serialLength, &head_size_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &ban_fp16_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new QkvToContextPluginDynamic(hidden_, head_number_, head_size_,
                                         scale_, ban_fp16_);
  }

  const char* getPluginType() const override { return "qkv_to_context_plugin"; }
  int getNbOutputs() const override { return 1; }
  int initialize() override;

  size_t getSerializationSize() const override {
    return 3 * sizeof(int) + sizeof(float) + sizeof(bool);
  }
  void serialize(void* buffer) const override {
    SerializeValue(&buffer, hidden_);
    SerializeValue(&buffer, head_number_);
    SerializeValue(&buffer, head_size_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, ban_fp16_);
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
  int hidden_;
  int head_number_;
  int head_size_;
  float scale_;
  bool ban_fp16_;
};

class QkvToContextPluginV2Creator : public nvinfer1::IPluginCreator {
 public:
  QkvToContextPluginV2Creator() {
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "hidden_", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "head_number_", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "head_size_", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "scale_", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "ban_fp16_", nullptr, nvinfer1::PluginFieldType::kINT8, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
  }
  const char* getPluginName() const override {
    return "qkv_to_context_pluginpaddle_trt";
  }

  const char* getPluginVersion() const override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &mFieldCollection;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    int hidden_, head_number_, head_size_;
    float scale_;
    bool ban_fp16_;
    const nvinfer1::PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
      const char* attrName = fields[i].name;
      if (!strcmp(attrName, "hidden_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
        hidden_ = *(static_cast<const int*>(fields[i].data));
      }
      if (!strcmp(attrName, "head_number_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
        head_number_ = *(static_cast<const int*>(fields[i].data));
      }
      if (!strcmp(attrName, "head_size_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
        head_size_ = *(static_cast<const int*>(fields[i].data));
      }
      if (!strcmp(attrName, "scale_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
        scale_ = *(static_cast<const float*>(fields[i].data));
      }
      if (!strcmp(attrName, "ban_fp16_")) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kINT8);
        ban_fp16_ = *(static_cast<const bool*>(fields[i].data));
      }
    }

    return new QkvToContextPluginDynamic(hidden_, head_number_, head_size_,
                                         scale_, ban_fp16_);
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) override {
    auto plugin = new QkvToContextPluginDynamic(serialData, serialLength);
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
REGISTER_TENSORRT_PLUGIN(QkvToContextPluginV2Creator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
