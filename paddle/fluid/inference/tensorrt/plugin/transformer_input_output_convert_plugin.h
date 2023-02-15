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
#include <cassert>
#include <string>

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class TransformerInputConvertPlugin : public DynamicPluginTensorRT {
 public:
  TransformerInputConvertPlugin() {}

  TransformerInputConvertPlugin(void const* serial_data, size_t serial_length) {
  }

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    TransformerInputConvertPlugin* ptr = new TransformerInputConvertPlugin();
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "transformer_input_convert_plugin";
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 4; }

  int initialize() TRT_NOEXCEPT { return 0; }
  void terminate() TRT_NOEXCEPT;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
                                          const nvinfer1::DimsExprs* inputs,
                                          int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* outputs,
                       int nbOutputs) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  void attachToContext(cudnnContext* cudnnContext,
                       cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* gpuAllocator)
      TRT_NOEXCEPT override;

  void detachFromContext() TRT_NOEXCEPT override;

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

 protected:
  size_t getSerializationSize() const TRT_NOEXCEPT override { return 0; }

  void serialize(void* buffer) const TRT_NOEXCEPT override {}
};

class TransformerInputConvertPluginCreator : public nvinfer1::IPluginCreator {
 public:
  TransformerInputConvertPluginCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "transformer_input_convert_plugin";
  }
  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* plugin_field)
      TRT_NOEXCEPT override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    TransformerInputConvertPlugin* obj =
        new TransformerInputConvertPlugin(serial_data, serial_length);
    obj->setPluginNamespace(name);
    return obj;
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
};

class TransformerOutputConvertPlugin : public DynamicPluginTensorRT {
 public:
  TransformerOutputConvertPlugin() {}

  TransformerOutputConvertPlugin(void const* serial_data,
                                 size_t serial_length) {}

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    TransformerOutputConvertPlugin* ptr = new TransformerOutputConvertPlugin();
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "transformer_output_convert_plugin";
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

  int initialize() TRT_NOEXCEPT { return 0; }
  void terminate() TRT_NOEXCEPT;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
                                          const nvinfer1::DimsExprs* inputs,
                                          int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* outputs,
                       int nbOutputs) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  void attachToContext(cudnnContext* cudnnContext,
                       cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* gpuAllocator)
      TRT_NOEXCEPT override;

  void detachFromContext() TRT_NOEXCEPT override;

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

 protected:
  size_t getSerializationSize() const TRT_NOEXCEPT override { return 0; }

  void serialize(void* buffer) const TRT_NOEXCEPT override {}
};

class TransformerOutputConvertPluginCreator : public nvinfer1::IPluginCreator {
 public:
  TransformerOutputConvertPluginCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "transformer_output_convert_plugin";
  }
  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* plugin_field)
      TRT_NOEXCEPT override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         void const* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    TransformerOutputConvertPlugin* obj =
        new TransformerOutputConvertPlugin(serial_data, serial_length);
    obj->setPluginNamespace(name);
    return obj;
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
};

REGISTER_TRT_PLUGIN_V2(TransformerInputConvertPluginCreator);
REGISTER_TRT_PLUGIN_V2(TransformerOutputConvertPluginCreator);
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
