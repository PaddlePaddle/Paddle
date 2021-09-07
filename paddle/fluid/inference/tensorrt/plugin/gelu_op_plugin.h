// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdio.h>
#include <cassert>
#include <string>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class GeluPlugin : public PluginTensorRT {
 public:
  explicit GeluPlugin(const bool with_fp16) { with_fp16_ = with_fp16; }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  GeluPlugin(void const* serial_data, size_t serial_length) {
    deserializeBase(serial_data, serial_length);
  }

  ~GeluPlugin() {}
  GeluPlugin* clone() const TRT_NOEXCEPT override {
    return new GeluPlugin(with_fp16_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "gelu_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override { return 0; }
  bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format)
      const TRT_NOEXCEPT override;
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nb_input_dims) TRT_NOEXCEPT override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batch_size, const void* const* inputs, void** outputs,
#else
  int enqueue(int batch_size, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return getBaseSerializationSize();
  }

  // TRT will call this func  to serialize the configuration of TRT
  // It should not be called by users.
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
  }
};

class GeluPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "gelu_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    return new GeluPlugin(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(GeluPluginCreator);

#if IS_TRT_VERSION_GE(6000)
class GeluPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit GeluPluginDynamic(const bool with_fp16) { with_fp16_ = with_fp16; }
  GeluPluginDynamic(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &with_fp16_);
  }

  ~GeluPluginDynamic() {}
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new GeluPluginDynamic(with_fp16_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "gelu_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override { return 0; }

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(with_fp16_);
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
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

  void destroy() TRT_NOEXCEPT override { delete this; }
};

class GeluPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "gelu_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    auto plugin = new GeluPluginDynamic(serial_data, serial_length);
    return plugin;
  }
};
REGISTER_TRT_PLUGIN_V2(GeluPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
