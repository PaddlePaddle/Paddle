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
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class PReluPlugin : public PluginTensorRT {
  std::vector<float> weight_;
  float* p_gpu_weight_;
  std::string mode_;

 public:
  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return getBaseSerializationSize() + SerializedSize(mode_.c_str()) +
           SerializedSize(weight_);
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
    SerializeValue(&buffer, weight_);
    SerializeValue(&buffer, mode_.c_str());
  }

  PReluPlugin(const float* weight, const int weight_num,
              std::string const& mode)
      : mode_(mode) {
    weight_.resize(weight_num);
    std::copy(weight, weight + weight_num, weight_.data());
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  PReluPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &weight_);
    const char* prelu_mode;
    DeserializeValue(&serialData, &serialLength, &prelu_mode);
    mode_ = std::string(prelu_mode);
  }
  ~PReluPlugin() {}
  int initialize() TRT_NOEXCEPT override;
  void terminate() TRT_NOEXCEPT override;

  PReluPlugin* clone() const TRT_NOEXCEPT override {
    auto* ptr = new PReluPlugin(weight_.data(), weight_.size(), mode_);
    ptr->p_gpu_weight_ = p_gpu_weight_;
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "prelu_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) TRT_NOEXCEPT override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
#else
  int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;
};

class PReluPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "prelu_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    return new PReluPlugin(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(PReluPluginCreator);

#if IS_TRT_VERSION_GE(6000)
class PReluPluginDynamic : public DynamicPluginTensorRT {
 public:
  PReluPluginDynamic(const float* weight, const int weight_num,
                     std::string const& mode)
      : mode_(mode) {
    weight_.resize(weight_num);
    std::copy(weight, weight + weight_num, weight_.data());
  }

  PReluPluginDynamic(void const* serialData, size_t serialLength);
  ~PReluPluginDynamic() {}
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    auto ptr = new PReluPluginDynamic(weight_.data(), weight_.size(), mode_);
    ptr->p_gpu_weight_ = p_gpu_weight_;
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "prelu_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override;
  void terminate() TRT_NOEXCEPT override;

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
  std::vector<float> weight_;
  float* p_gpu_weight_;
  std::string mode_;
};
#endif

class PReluPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "prelu_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    return new PReluPluginDynamic(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(PReluPluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
