/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#if IS_TRT_VERSION_LT(8000)
class ElementWisePlugin : public PluginTensorRT {
 public:
  ElementWisePlugin(std::string type, nvinfer1::Dims const& dims_x,
                    nvinfer1::Dims const& dims_y, int axis)
      : type_(type),
        dims_x_(dims_x),
        dims_y_(dims_y),
        axis_(axis),
        prev_size_(1),
        midd_size_(1),
        post_size_(1) {}

  ElementWisePlugin(void const* serial_data, size_t serial_length) {
    deserializeBase(serial_data, serial_length);
    const char* elementwise_type;
    DeserializeValue(&serial_data, &serial_length, &elementwise_type);
    type_ = std::string(elementwise_type);
    DeserializeValue(&serial_data, &serial_length, &axis_);
    DeserializeValue(&serial_data, &serial_length, &dims_x_);
    DeserializeValue(&serial_data, &serial_length, &dims_y_);
  }

  ElementWisePlugin* clone() const override {
    // return new ElementWisePlugin(dims_x_, dims_y_, axis_);
    return nullptr;
  }

  const char* getPluginType() const override { return "elementwise_plugin"; }

  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* input_dims,
                                     int num_inputs) override;

  int initialize() override;

  // execute the layer
  int enqueue(int batch_size, const void* const* inputs, void** outputs,
              void* workspace, cudaStream_t stream);

 protected:
  size_t getSerializationSize() override {
    return SerializedSize(getPluginType()) + SerializedSize(axis_) +
           SerializedSize(dims_x_) + SerializedSize(dims_y_) +
           getBaseSerializationSize();
  }

  void serialize(void* buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, type_.c_str());
    SerializeValue(&buffer, axis_);
    SerializeValue(&buffer, dims_x_);
    SerializeValue(&buffer, dims_y_);
  }

  std::string type_;
  nvinfer1::Dims dims_x_;
  nvinfer1::Dims dims_y_;
  int axis_;
  int prev_size_;
  int midd_size_;
  int post_size_;
};
#endif

#if IS_TRT_VERSION_GE(6000)
class ElementwisePluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit ElementwisePluginDynamic(const std::string& type, int axis)
      : type_(type), axis_(axis) {}
  ElementwisePluginDynamic(void const* serialData, size_t serialLength) {
    const char* elementwise_type;
    DeserializeValue(&serialData, &serialLength, &elementwise_type);
    type_ = std::string(elementwise_type);
    DeserializeValue(&serialData, &serialLength, &axis_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new ElementwisePluginDynamic(type_, axis_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "elementwise_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;
  void serialize(void* buffer) const TRT_NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT override;

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
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* inputTypes,
      int nbInputs) const TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

 private:
  std::string type_;
  int axis_;
};

class ElementwisePluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  ElementwisePluginDynamicCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "elementwise_plugin";
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
    auto plugin = new ElementwisePluginDynamic(serial_data, serial_length);
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

REGISTER_TRT_PLUGIN_V2(ElementwisePluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
