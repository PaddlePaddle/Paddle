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
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)
class TransformerInputConvertPlugin : public DynamicPluginTensorRT {
 public:
  TransformerInputConvertPlugin() {}

  TransformerInputConvertPlugin(void const* serial_data, size_t serial_length) {
    DeserializeValue(&serial_data, &serial_length, &transB_);
    DeserializeValue(&serial_data, &serial_length, &transA_);
    DeserializeValue(&serial_data, &serial_length, &alpha_);
    DeserializeValue(&serial_data, &serial_length, &alpha_scale_);
    DeserializeValue(&serial_data, &serial_length, &alpha_one_);
    DeserializeValue(&serial_data, &serial_length, &alpha_zero_);
    DeserializeValue(&serial_data, &serial_length, &cublas_);
    DeserializeValue(&serial_data, &serial_length, &Atransform_);
    DeserializeValue(&serial_data, &serial_length, &Btransform_);
    DeserializeValue(&serial_data, &serial_length, &Ctransform_);
    DeserializeValue(&serial_data, &serial_length, &type_);
  }

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    TransformerInputConvertPlugin* ptr = new TransformerInputConvertPlugin();
    ptr->setPluginNamespace(this->getPluginNamespace());
    ptr->alpha_scale_ = alpha_scale_;
    ptr->alpha_one_ = alpha_one_;
    ptr->alpha_zero_ = alpha_zero_;
    ptr->cublas_ = cublas_;
    ptr->Atransform_ = Atransform_;
    ptr->Btransform_ = Btransform_;
    ptr->Ctransform_ = Ctransform_;
    ptr->type_ = type_;
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "matmul_int8_dynamic_plugin";
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

  int initialize() TRT_NOEXCEPT { return 0; }
  void terminate() TRT_NOEXCEPT;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT override;

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

  void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext,
                       nvinfer1::IGpuAllocator* gpuAllocator)
      TRT_NOEXCEPT override;

  void detachFromContext() TRT_NOEXCEPT override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* inputTypes,
      int nbInputs) const TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }

 protected:
  bool transB_;
  bool transA_;
  float alpha_;
  void *alpha_scale_{nullptr}, *alpha_one_{nullptr}, *alpha_zero_{nullptr};
  cublasLtHandle_t cublas_{nullptr};
  nvinfer1::DataType type_;
  int8_t *Atransform_{nullptr}, *Btransform_{nullptr}, *Ctransform_{nullptr};
  std::string name_space_;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(transB_) + SerializedSize(transA_) +
           SerializedSize(alpha_) + SerializedSize(alpha_scale_) +
           SerializedSize(alpha_one_) + SerializedSize(alpha_zero_) +
           SerializedSize(Atransform_) + SerializedSize(Btransform_) +
           SerializedSize(Ctransform_) + SerializedSize(cublas_) +
           SerializedSize(type_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, transB_);
    SerializeValue(&buffer, transA_);
    SerializeValue(&buffer, alpha_);
    SerializeValue(&buffer, alpha_scale_);
    SerializeValue(&buffer, alpha_one_);
    SerializeValue(&buffer, alpha_zero_);
    SerializeValue(&buffer, Atransform_);
    SerializeValue(&buffer, Btransform_);
    SerializeValue(&buffer, Ctransform_);
    SerializeValue(&buffer, cublas_);
    SerializeValue(&buffer, type_);
  }
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

  nvinfer1::IPluginV2* createPlugin(const char* name,
                                    const nvinfer1::PluginFieldCollection* fc)
      TRT_NOEXCEPT override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, void const* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
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
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};
REGISTER_TRT_PLUGIN_V2(TransformerInputConvertPluginCreator);
#endif
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
