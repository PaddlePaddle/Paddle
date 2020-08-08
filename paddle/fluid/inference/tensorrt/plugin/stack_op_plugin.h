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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

#if IS_TRT_VERSION_GE(6000)
class StackPluginDynamic : public DynamicPluginTensorRT {
 public:
  StackPluginDynamic(int axis, int num_stack)
      : axis_(axis), num_stack_(num_stack) {
    init();
  }

  StackPluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &axis_);
    DeserializeValue(&serialData, &serialLength, &num_stack_);
    init();
  }

  ~StackPluginDynamic() {}
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new StackPluginDynamic(axis_, num_stack_);
  }

  void init() {
    int device_id;
    cudaGetDevice(&device_id);
    in_ptr_tensor_.Resize({num_stack_});
    in_ptr_gpu_ =
        in_ptr_tensor_.mutable_data<int64_t>(platform::CUDAPlace(device_id));
  }

  const char* getPluginType() const override { return "stack_plugin"; }
  int getNbOutputs() const override { return 1; }
  int initialize() override { return 0; }

  size_t getSerializationSize() const override {
    size_t serialize_size = 0;

    serialize_size += SerializedSize(axis_);
    serialize_size += SerializedSize(num_stack_);

    return serialize_size;
  }

  void serialize(void* buffer) const override {
    SerializeValue(&buffer, axis_);
    SerializeValue(&buffer, num_stack_);
  }

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) override;

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
  int axis_;
  int num_stack_;
  framework::Tensor in_ptr_tensor_;
  int64_t* in_ptr_gpu_;
};

class StackPluginV2Creator : public nvinfer1::IPluginCreator {
 public:
  StackPluginV2Creator() {}
  const char* getPluginName() const override { return "stack_plugin"; }

  const char* getPluginVersion() const override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length) override {
    auto plugin = new StackPluginDynamic(serial_data, serial_length);
    return plugin;
  }

  void setPluginNamespace(const char* lib_namespace) override {
    plugin_namespace_ = lib_namespace;
  }

  const char* getPluginNamespace() const override {
    return plugin_namespace_.c_str();
  }

 private:
  std::string plugin_namespace_;
  std::string plugin_name_;
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
  std::vector<nvinfer1::PluginField> plugin_attributes_;
};
REGISTER_TRT_PLUGIN_V2(StackPluginV2Creator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
