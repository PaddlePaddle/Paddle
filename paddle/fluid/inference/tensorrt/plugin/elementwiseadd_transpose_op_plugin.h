/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class ElementwiseAddTransposePluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit ElementwiseAddTransposePluginDynamic(int axis,
                                                std::vector<int> output_shape)
      : axis_(axis), output_shape_(output_shape) {}
  ElementwiseAddTransposePluginDynamic(void const* serialData,
                                       size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &axis_);
    DeserializeValue(&serialData, &serialLength, &output_shape_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new ElementwiseAddTransposePluginDynamic(axis_, output_shape_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "elementwise_add_transpose_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void* buffer) const TRT_NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

 private:
  int axis_;
  std::vector<int> output_shape_;
};

class ElementwiseAddTransposePluginDynamicCreator
    : public nvinfer1::IPluginCreator {
 public:
  ElementwiseAddTransposePluginDynamicCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "elementwise_add_transpose_plugin_dynamic";
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
    auto plugin =
        new ElementwiseAddTransposePluginDynamic(serial_data, serial_length);
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
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};

REGISTER_TRT_PLUGIN_V2(ElementwiseAddTransposePluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
