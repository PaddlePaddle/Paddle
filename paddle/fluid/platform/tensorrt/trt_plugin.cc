// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/tensorrt/trt_plugin.h"

namespace paddle::platform {

inline void Serialize(void*& buffer,  // NOLINT
                      const std::vector<nvinfer1::Dims>& input_dims,
                      nvinfer1::DataType data_type,
                      nvinfer1::PluginFormat data_format,
                      bool with_fp16) {
  SerializeValue(&buffer, input_dims);
  SerializeValue(&buffer, data_type);
  SerializeValue(&buffer, data_format);
  SerializeValue(&buffer, with_fp16);
}

inline void Deserialize(void const*& serial_data,  // NOLINT
                        size_t& serial_length,     // NOLINT
                        std::vector<nvinfer1::Dims>* input_dims,
                        nvinfer1::DataType* data_type,
                        nvinfer1::PluginFormat* data_format,
                        bool* with_fp16) {
  DeserializeValue(&serial_data, &serial_length, input_dims);
  DeserializeValue(&serial_data, &serial_length, data_type);
  DeserializeValue(&serial_data, &serial_length, data_format);
  DeserializeValue(&serial_data, &serial_length, with_fp16);
}

inline size_t SerializeSize(const std::vector<nvinfer1::Dims>& input_dims,
                            nvinfer1::DataType data_type,
                            nvinfer1::PluginFormat data_format,
                            bool with_fp16) {
  return (SerializedSize(input_dims) + SerializedSize(data_type) +
          SerializedSize(data_format) + SerializedSize(with_fp16));
}

void PluginTensorRT::serializeBase(void*& buffer) const {
  Serialize(buffer, input_dims_, data_type_, data_format_, with_fp16_);
}

void PluginTensorRT::deserializeBase(void const*& serial_data,
                                     size_t& serial_length) {
  Deserialize(serial_data,
              serial_length,
              &input_dims_,
              &data_type_,
              &data_format_,
              &with_fp16_);
}

size_t PluginTensorRT::getBaseSerializationSize() const {
  return SerializeSize(input_dims_, data_type_, data_format_, with_fp16_);
}

bool PluginTensorRT::supportsFormat(
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const TRT_NOEXCEPT {
  return ((type == nvinfer1::DataType::kFLOAT) &&
          (format == nvinfer1::PluginFormat::kLINEAR));
}

void PluginTensorRT::configureWithFormat(const nvinfer1::Dims* input_dims,
                                         int num_inputs,
                                         const nvinfer1::Dims* output_dims,
                                         int num_outputs,
                                         nvinfer1::DataType type,
                                         nvinfer1::PluginFormat format,
                                         int max_batch_size) TRT_NOEXCEPT {
  data_type_ = type;
  data_format_ = format;
  input_dims_.assign(input_dims, input_dims + num_inputs);
}

void PluginTensorRTV2Ext::serializeBase(void*& buffer) const {
  Serialize(buffer, input_dims_, data_type_, data_format_, with_fp16_);
}

void PluginTensorRTV2Ext::deserializeBase(void const*& serial_data,
                                          size_t& serial_length) {
  Deserialize(serial_data,
              serial_length,
              &input_dims_,
              &data_type_,
              &data_format_,
              &with_fp16_);
}

size_t PluginTensorRTV2Ext::getBaseSerializationSize() const {
  return SerializeSize(input_dims_, data_type_, data_format_, with_fp16_);
}

void PluginTensorRTV2Ext::configurePlugin(
    const nvinfer1::Dims* input_dims,
    int32_t nb_inputs,
    const nvinfer1::Dims* output_dims,
    int32_t nb_outputs,
    const nvinfer1::DataType* input_types,
    const nvinfer1::DataType* output_types,
    const bool* input_is_broadcast,
    const bool* output_is_broadcast,
    nvinfer1::PluginFormat float_format,
    int32_t max_batch_size) TRT_NOEXCEPT {
  input_dims_.assign(input_dims, input_dims + nb_inputs);
  data_format_ = float_format;
  data_type_ = input_types[0];
}

const nvinfer1::PluginFieldCollection* TensorRTPluginCreator::getFieldNames()
    TRT_NOEXCEPT {
  return &field_collection_;
}

nvinfer1::IPluginV2* TensorRTPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT {
  return nullptr;
}

void TensorRTPluginCreator::setPluginNamespace(const char* lib_namespace)
    TRT_NOEXCEPT {
  plugin_namespace_ = lib_namespace;
}

const char* TensorRTPluginCreator::getPluginNamespace() const TRT_NOEXCEPT {
  return plugin_namespace_.c_str();
}

}  // namespace paddle::platform
