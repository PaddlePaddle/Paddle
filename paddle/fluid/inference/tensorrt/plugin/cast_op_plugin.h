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
#include <stdio.h>

#include <cassert>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class CastPlugin : public PluginTensorRTV2Ext {
 public:
  CastPlugin(int intype, int outtype, bool fp16)
      : intype_(intype), outtype_(outtype), fp16_(fp16) {}

  CastPlugin(void const* serial_data, size_t serial_length) {
    deserializeBase(serial_data, serial_length);
    DeserializeValue(&serial_data, &serial_length, &intype_);
    DeserializeValue(&serial_data, &serial_length, &outtype_);
    DeserializeValue(&serial_data, &serial_length, &fp16_);
  }

  nvinfer1::IPluginV2Ext* clone() const TRT_NOEXCEPT override {
    CastPlugin* ptr = new CastPlugin(intype_, outtype_, fp16_);
    ptr->setPluginNamespace(this->getPluginNamespace());
    ptr->data_format_ = data_format_;
    ptr->data_type_ = data_type_;
    ptr->input_dims_ = input_dims_;
    return ptr;
  }

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const
      TRT_NOEXCEPT override {
    if (outtype_ == 0) {
      return nvinfer1::DataType::kBOOL;
    } else if (outtype_ == 2) {
      return nvinfer1::DataType::kINT32;
    } else {
      return nvinfer1::DataType::kFLOAT;
    }
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "cast_plugin_v2ext";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* input_dims,
                                     int num_inputs) TRT_NOEXCEPT override;

  bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format)
      const TRT_NOEXCEPT override {
    if (fp16_) {
      return type == nvinfer1::DataType::kFLOAT ||
             type == nvinfer1::DataType::kINT32 ||
             type == nvinfer1::DataType::kBOOL;

    } else {
      return type == nvinfer1::DataType::kFLOAT ||
             type == nvinfer1::DataType::kINT32 ||
             type == nvinfer1::DataType::kBOOL;
    }
  }

  int initialize() TRT_NOEXCEPT override;
  void terminate() TRT_NOEXCEPT override;
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
              cudaStream_t stream) TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

 protected:
  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(intype_) + SerializedSize(outtype_) +
           SerializedSize(fp16_) + getBaseSerializationSize();
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
    SerializeValue(&buffer, intype_);
    SerializeValue(&buffer, outtype_);
    SerializeValue(&buffer, fp16_);
  }

 private:
  int intype_;
  int outtype_;
  bool fp16_;
};

class CastPluginCreator : public nvinfer1::IPluginCreator {
 public:
  CastPluginCreator() {}
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "cast_plugin_v2ext";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc)
      TRT_NOEXCEPT override {
    // not implemented
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    auto plugin = new CastPlugin(serial_data, serial_length);
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

REGISTER_TRT_PLUGIN_V2(CastPluginCreator);

#if IS_TRT_VERSION_GE(6000)
class CastPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit CastPluginDynamic(int intype, int outtype, bool fp16) {
    intype_ = intype;
    outtype_ = outtype;
    fp16_ = fp16;
  }
  CastPluginDynamic(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &intype_);
    DeserializeValue(&serial_data, &serial_length, &outtype_);
  }

  ~CastPluginDynamic() {}
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new CastPluginDynamic(intype_, outtype_, fp16_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "cast_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override { return 0; }

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(intype_) + SerializedSize(outtype_) +
           SerializedSize(fp16_);
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, intype_);
    SerializeValue(&buffer, outtype_);
    SerializeValue(&buffer, fp16_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT override;

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
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  void destroy() TRT_NOEXCEPT override { delete this; }
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* input_types,
                                       int nb_inputs) const
      TRT_NOEXCEPT override;

 private:
  int intype_;
  int outtype_;
  bool fp16_;
};

class CastPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "cast_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    auto plugin = new CastPluginDynamic(serial_data, serial_length);
    return plugin;
  }
};
REGISTER_TRT_PLUGIN_V2(CastPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
