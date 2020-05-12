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

#include <NvInfer.h>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/inference/tensorrt/helper.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler.h"

DECLARE_bool(profile);

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class PluginTensorRT;

typedef std::function<PluginTensorRT*(const void*, size_t)>
    PluginDeserializeFunc;

typedef std::function<PluginTensorRT*(void)> PluginConstructFunc;

class PluginTensorRT : public nvinfer1::IPluginExt {
 public:
  PluginTensorRT() {}
  // It was used for TensorRT deserialization.
  // It should not be called by users.
  PluginTensorRT(const void* serialized_data, size_t length) {}
  virtual ~PluginTensorRT() {}

  nvinfer1::Dims const& getInputDims(int index) const {
    return input_dims_.at(index);
  }
  size_t getMaxBatchSize() const { return max_batch_size_; }
  nvinfer1::DataType getDataType() const { return data_type_; }
  nvinfer1::PluginFormat getDataFormat() const { return data_format_; }
  virtual const char* getPluginVersion() const { return "1"; }

  void AddInput(nvinfer1::ITensor* input) { inputs_.push_back(input); }
  std::vector<nvinfer1::ITensor*>& GetInputs() { return inputs_; }

  virtual nvinfer1::IPluginExt* clone() const = 0;
  virtual const char* getPluginType() const = 0;

  // Following functions are inherit from nvinfer1::IPluginExt
  // Get the number of outputs from the layer
  int getNbOutputs() const { return 1; }
  // Get the dimension of an output tensor
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims* input_dims,
                                             int num_inputs) = 0;
  // Find the workspace size required by the layer
  size_t getWorkspaceSize(int) const override { return 0; }

  // Initialize the layer for execution.
  // This is called when the engine is created.
  int initialize() override { return 0; }
  // Shutdown the layer. This is called when the engine is destroyed
  void terminate() override {}
  // Execute the layer
  virtual int enqueue(int batch_size, const void* const* inputs, void** outputs,
                      void* workspace, cudaStream_t stream) = 0;

  // Find the size of the serialization buffer required
  virtual size_t getSerializationSize() = 0;
  // Serialize the layer config to buffer.
  // TensorRT will call this func to serialize the configuration of TensorRT
  // engine. It should not be called by users.
  virtual void serialize(void* buffer) = 0;

  // Check format support. The default is FLOAT32 and NCHW.
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override;
  // Configure the layer
  void configureWithFormat(const nvinfer1::Dims* input_dims, int num_inputs,
                           const nvinfer1::Dims* output_dims, int num_outputs,
                           nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int max_batch_size) override;

 protected:
  // Deserialize input_dims, max_batch_size, data_type, data_format
  void deserializeBase(void const*& serial_data,  // NOLINT
                       size_t& serial_length);    // NOLINT
  size_t getBaseSerializationSize();
  // Serialize input_dims, max_batch_size, data_type, data_format
  void serializeBase(void*& buffer);  // NOLINT

  std::vector<nvinfer1::Dims> input_dims_;
  size_t max_batch_size_;
  nvinfer1::DataType data_type_;
  nvinfer1::PluginFormat data_format_;

  std::vector<nvinfer1::ITensor*> inputs_;
};

#if IS_TRT_VERSION_GE(6000)
class DynamicPluginTensorRT : public nvinfer1::IPluginV2DynamicExt {
 public:
  DynamicPluginTensorRT() {}
  DynamicPluginTensorRT(const void* serialized_data, size_t length) {}

  // The Func in IPluginExt or IpluginExtV2
  virtual const char* getPluginVersion() const { return "1"; }
  virtual const char* getPluginType() const = 0;
  int getNbOutputs() const { return 1; }
  int initialize() override { return 0; }
  void terminate() override{};

  virtual size_t getSerializationSize() const = 0;
  virtual void serialize(void* buffer) const = 0;

  // The Func in IPluginV2
  nvinfer1::IPluginV2DynamicExt* clone() const = 0;
  virtual nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) = 0;  // NOLINT

  virtual bool supportsFormatCombination(
      int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
      int nb_outputs) = 0;

  virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                               int nb_inputs,
                               const nvinfer1::DynamicPluginTensorDesc* out,
                               int nb_outputs) = 0;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nb_inputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nb_outputs) const override {
    return 0;
  }

  virtual int enqueue(const nvinfer1::PluginTensorDesc* input_desc,
                      const nvinfer1::PluginTensorDesc* output_desc,
                      const void* const* inputs, void* const* outputs,
                      void* workspace, cudaStream_t stream) = 0;

  virtual nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* input_types,
      int nb_inputs) const = 0;
  void setPluginNamespace(const char* plugin_namespace) override {
    name_space_ = plugin_namespace;
  }
  const char* getPluginNamespace() const override {
    return name_space_.c_str();
  }
  virtual void destroy() = 0;

 protected:
  void deserializeBase(void const*& serial_data,  // NOLINT
                       size_t& serial_length);    // NOLINT
  size_t getBaseSerializationSize() const;
  void serializeBase(void*& buffer) const;  // NOLINT

 private:
  std::string name_space_{"paddle_trt"};
  std::string plugin_base_{"plugin_dynamic"};
};
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
