
// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class custom_op_plugin : public nvinfer1::IPluginV2 {
 public:
  explicit custom_op_plugin(float float_attr) { float_attr_ = float_attr; }

  custom_op_plugin(const void* buffer, size_t length) {
    DeserializeValue(&buffer, &length, &float_attr_);
  }

  size_t getSerializationSize() const noexcept override {
    return SerializedSize(float_attr_);
  }

  void serialize(void* buffer) const noexcept override {
    SerializeValue(&buffer, float_attr_);
  }

  nvinfer1::IPluginV2* clone() const noexcept override {
    return new custom_op_plugin(float_attr_);
  }

  ~custom_op_plugin() override = default;

  const char* getPluginType() const noexcept override {
    return "custom_op_paddle_trt_plugin";
  }

  const char* getPluginVersion() const noexcept override { return "1"; }

  int getNbOutputs() const noexcept override { return 1; }

  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* inputs,
                                     int nbInputDims) noexcept override {
    return inputs[0];
  }

  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const noexcept override {
    return true;
  }

  void configureWithFormat(nvinfer1::Dims const* inputDims,
                           int32_t nbInputs,
                           nvinfer1::Dims const* outputDims,
                           int32_t nbOutputs,
                           nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int32_t maxBatchSize) noexcept override {}

  int initialize() noexcept override { return 0; }

  void terminate() noexcept override {}

  size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
    return 0;
  }

#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batch_size,
              const void* const* inputs,
              void** outputs,
#else
  int enqueue(int batch_size,
              const void* const* inputs,
              void* const* outputs,
#endif
              void* workspace,
              cudaStream_t stream) noexcept override {
    return 0;
  }

  void destroy() noexcept override { delete this; }

  void setPluginNamespace(const char* libNamespace) noexcept override {
    namespace_ = libNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

 private:
  float float_attr_;
  std::string namespace_;
};

class custom_op_plugin_creator : public nvinfer1::IPluginCreator {
 public:
  custom_op_plugin_creator() {}

  ~custom_op_plugin_creator() override = default;

  const char* getPluginName() const noexcept override {
    return "custom_op_paddle_trt_plugin";
  }

  const char* getPluginVersion() const noexcept override { return "1"; }

  void setPluginNamespace(const char* pluginNamespace) noexcept override {
    plugin_namespace_ = pluginNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return plugin_namespace_.c_str();
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return nullptr;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override {
    PADDLE_ENFORCE_EQ(
        fc->nbFields,
        7,
        phi::errors::InvalidArgument("fc->nbFields is invalid. "
                                     "Expected 7, but received %d.",
                                     fc->nbFields));
    // float_attr
    auto attr_field = (fc->fields)[0];
    PADDLE_ENFORCE_EQ(
        attr_field.type,
        nvinfer1::PluginFieldType::kFLOAT32,
        phi::errors::InvalidArgument("The attr_field type must be "
                                     "nvinfer1::PluginFieldType::kFLOAT32"));
    PADDLE_ENFORCE_EQ(attr_field.length,
                      1,
                      phi::errors::InvalidArgument(
                          "The length of attr_field is invalid. "
                          "Expected attr_field.length=1, but received %d.",
                          attr_field.length));
    float float_value = (reinterpret_cast<const float*>(attr_field.data))[0];
    PADDLE_ENFORCE_EQ(
        float_value,
        1.0,
        phi::errors::InvalidArgument("float_value is invalid. "
                                     "Expected 1.0, but received %f.",
                                     float_value));

    // int_attr
    attr_field = (fc->fields)[1];
    PADDLE_ENFORCE_EQ(
        attr_field.type,
        nvinfer1::PluginFieldType::kINT32,
        phi::errors::InvalidArgument("The attr_field type must be "
                                     "nvinfer1::PluginFieldType::kINT32"));
    PADDLE_ENFORCE_EQ(attr_field.length,
                      1,
                      phi::errors::InvalidArgument(
                          "The length of attr_field is invalid. "
                          "Expected attr_field.length=1, but received %d.",
                          attr_field.length));
    int int_value = (reinterpret_cast<const int*>(attr_field.data))[0];
    PADDLE_ENFORCE_EQ(
        int_value,
        1,
        phi::errors::InvalidArgument("int_value is invalid. "
                                     "Expected 1, but received %d.",
                                     int_value));

    // bool_attr
    attr_field = (fc->fields)[2];
    PADDLE_ENFORCE_EQ(
        attr_field.type,
        nvinfer1::PluginFieldType::kINT32,
        phi::errors::InvalidArgument("The attr_field type must be "
                                     "nvinfer1::PluginFieldType::kINT32"));
    PADDLE_ENFORCE_EQ(attr_field.length,
                      1,
                      phi::errors::InvalidArgument(
                          "The length of attr_field is invalid. "
                          "Expected attr_field.length=1, but received %d.",
                          attr_field.length));
    int bool_value = (reinterpret_cast<const int*>(attr_field.data))[0];
    PADDLE_ENFORCE_EQ(
        bool_value,
        1,
        phi::errors::InvalidArgument("bool_value is invalid. "
                                     "Expected 1, but received %d.",
                                     bool_value));

    // string_attr
    attr_field = (fc->fields)[3];
    PADDLE_ENFORCE_EQ(
        attr_field.type,
        nvinfer1::PluginFieldType::kCHAR,
        phi::errors::InvalidArgument("The attr_field type must be "
                                     "nvinfer1::PluginFieldType::kCHAR"));
    std::string expect_string_attr = "test_string_attr";
    PADDLE_ENFORCE_EQ(static_cast<size_t>(attr_field.length),
                      expect_string_attr.size() + 1,
                      phi::errors::InvalidArgument(
                          "The length of attr_field must be equal to "
                          "the size of expect_string_attr plus 1. "
                          "Expected %llu, but received %llu.",
                          static_cast<size_t>(expect_string_attr.size() + 1),
                          static_cast<size_t>(attr_field.length)));
    const char* receive_string_attr =
        reinterpret_cast<const char*>(attr_field.data);
    PADDLE_ENFORCE_EQ(
        expect_string_attr,
        std::string(receive_string_attr),
        phi::errors::InvalidArgument("The received string attribute '%s' "
                                     "does not match the expected value '%s'.",
                                     receive_string_attr,
                                     expect_string_attr.c_str()));

    // ints_attr
    attr_field = (fc->fields)[4];
    PADDLE_ENFORCE_EQ(
        attr_field.type,
        nvinfer1::PluginFieldType::kINT32,
        phi::errors::InvalidArgument("The attr_field type must be "
                                     "nvinfer1::PluginFieldType::kINT32"));
    PADDLE_ENFORCE_EQ(attr_field.length,
                      3,
                      phi::errors::InvalidArgument(
                          "The length of attr_field is invalid. "
                          "Expected attr_field.length=3, but received %d.",
                          attr_field.length));
    const int* ints_value = reinterpret_cast<const int*>(attr_field.data);
    PADDLE_ENFORCE_EQ(
        ints_value[0],
        1,
        phi::errors::InvalidArgument("ints_value[0] is invalid. "
                                     "Expected 1, but received %d.",
                                     ints_value[0]));
    PADDLE_ENFORCE_EQ(
        ints_value[1],
        2,
        phi::errors::InvalidArgument("ints_value[1] is invalid. "
                                     "Expected 2, but received %d.",
                                     ints_value[1]));
    PADDLE_ENFORCE_EQ(
        ints_value[2],
        3,
        phi::errors::InvalidArgument("ints_value[2] is invalid. "
                                     "Expected 3, but received %d.",
                                     ints_value[2]));

    // floats_attr
    attr_field = (fc->fields)[5];
    PADDLE_ENFORCE_EQ(
        attr_field.type,
        nvinfer1::PluginFieldType::kFLOAT32,
        phi::errors::InvalidArgument("The attr_field type must be "
                                     "nvinfer1::PluginFieldType::kFLOAT32"));
    PADDLE_ENFORCE_EQ(attr_field.length,
                      3,
                      phi::errors::InvalidArgument(
                          "The length of attr_field is invalid. "
                          "Expected attr_field.length=3, but received %d.",
                          attr_field.length));
    const float* floats_value = reinterpret_cast<const float*>(attr_field.data);
    PADDLE_ENFORCE_EQ(
        floats_value[0],
        1.0,
        phi::errors::InvalidArgument("floats_value[0] is invalid. "
                                     "Expected 1.0, but received %f.",
                                     floats_value[0]));
    PADDLE_ENFORCE_EQ(
        floats_value[1],
        2.0,
        phi::errors::InvalidArgument("floats_value[1] is invalid. "
                                     "Expected 2.0, but received %f.",
                                     floats_value[1]));
    PADDLE_ENFORCE_EQ(
        floats_value[2],
        3.0,
        phi::errors::InvalidArgument("floats_value[2] is invalid. "
                                     "Expected 3.0, but received %f.",
                                     floats_value[2]));

    // bools_attr
    attr_field = (fc->fields)[6];
    PADDLE_ENFORCE_EQ(
        attr_field.type,
        nvinfer1::PluginFieldType::kINT32,
        phi::errors::InvalidArgument("The attr_field type must be "
                                     "nvinfer1::PluginFieldType::kINT32"));
    PADDLE_ENFORCE_EQ(attr_field.length,
                      3,
                      phi::errors::InvalidArgument(
                          "The length of attr_field is invalid. "
                          "Expected attr_field.length=3, but received %d.",
                          attr_field.length));
    ints_value = reinterpret_cast<const int*>(attr_field.data);
    PADDLE_ENFORCE_EQ(
        ints_value[0],
        true,
        phi::errors::InvalidArgument("ints_value[0] is invalid. "
                                     "Expected true, but received false."));
    PADDLE_ENFORCE_EQ(
        ints_value[1],
        false,
        phi::errors::InvalidArgument("ints_value[1] is invalid. "
                                     "Expected false, but received true."));
    PADDLE_ENFORCE_EQ(
        ints_value[2],
        true,
        phi::errors::InvalidArgument("ints_value[2] is invalid. "
                                     "Expected true, but received false."));

    return new custom_op_plugin(float_value);
  }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name,
      const void* serialData,
      size_t serialLength) noexcept override {
    return new custom_op_plugin(serialData, serialLength);
  }

 private:
  std::string plugin_namespace_;
};

class custom_op_dynamic_plugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  explicit custom_op_dynamic_plugin(float float_attr)
      : float_attr_(float_attr) {}

  custom_op_dynamic_plugin(const void* buffer, size_t length) {
    DeserializeValue(&buffer, &length, &float_attr_);
  }

  ~custom_op_dynamic_plugin() override = default;

  const char* getPluginType() const noexcept override {
    return "custom_op_paddle_trt_dynamic_plugin";
  }

  const char* getPluginVersion() const noexcept override { return "1"; }

  int getNbOutputs() const noexcept override { return 1; }

  int initialize() noexcept override { return 0; }

  void terminate() noexcept override {}

  size_t getSerializationSize() const noexcept override {
    return SerializedSize(float_attr_);
  }

  void serialize(void* buffer) const noexcept override {
    SerializeValue(&buffer, float_attr_);
  }

  void destroy() noexcept override { delete this; }

  void setPluginNamespace(const char* libNamespace) noexcept override {
    namespace_ = libNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

  /*IPluginV2Ext method*/
  nvinfer1::DataType getOutputDataType(
      int32_t index,
      nvinfer1::DataType const* inputTypes,
      int32_t nbInputs) const noexcept override {
    return inputTypes[index];
  }

  /*IPluginV2DynamicExt method*/
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
    return new custom_op_dynamic_plugin(float_attr_);
  };

  nvinfer1::DimsExprs getOutputDimensions(
      int32_t outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int32_t nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override {
    return inputs[0];
  }

  bool supportsFormatCombination(int32_t pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    return true;
  }

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int32_t nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int32_t nbOutputs) noexcept override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int32_t nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int32_t nbOutputs) const noexcept override {
    return 0;
  }

  int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                  const nvinfer1::PluginTensorDesc* outputDesc,
                  const void* const* inputs,
                  void* const* outputs,
                  void* workspace,
                  cudaStream_t stream) noexcept override {
    return 0;
  }

 private:
  float float_attr_ = 0;
  std::string namespace_;
};

class custom_op_dynamic_plugin_creator : public nvinfer1::IPluginCreator {
 public:
  custom_op_dynamic_plugin_creator() {}

  ~custom_op_dynamic_plugin_creator() override = default;

  const char* getPluginName() const noexcept override {
    return "custom_op_paddle_trt_dynamic_plugin";
  }

  const char* getPluginVersion() const noexcept override { return "1"; }

  void setPluginNamespace(char const* pluginNamespace) noexcept override {
    plugin_namespace_ = pluginNamespace;
  }

  const char* getPluginNamespace() const noexcept override {
    return plugin_namespace_.c_str();
  }

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override {
    return nullptr;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) noexcept override {
    return new custom_op_dynamic_plugin(1.0);
  }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name,
      const void* serialData,
      size_t serialLength) noexcept override {
    return new custom_op_dynamic_plugin(serialData, serialLength);
  }

 private:
  std::string plugin_namespace_;
};

REGISTER_TRT_PLUGIN_V2(custom_op_plugin_creator);
REGISTER_TRT_PLUGIN_V2(custom_op_dynamic_plugin_creator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
