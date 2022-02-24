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
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class GroupNormPlugin : public PluginTensorRTV2Ext {
 private:
  std::string plugin_namespace_;
  float epsilon_;
  int groups_;
  std::vector<int> input_dims_;
  bool with_fp16_ = false;

  float* bn_scale_;
  float* bn_bias_;

  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t desc_, bn_desc_;

 public:
  explicit GroupNormPlugin(const float epsilon, const int groups)
      : epsilon_(epsilon), groups_(groups) {
    platform::dynload::cudnnCreate(&handle_);
    platform::dynload::cudnnCreateTensorDescriptor(&desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&bn_desc_);
  }

  GroupNormPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &epsilon_);
    DeserializeValue(&serialData, &serialLength, &groups_);
    DeserializeValue(&serialData, &serialLength, &input_dims_);
    DeserializeValue(&serialData, &serialLength, &with_fp16_);

    platform::dynload::cudnnCreate(&handle_);
    platform::dynload::cudnnCreateTensorDescriptor(&desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&bn_desc_);
  }

  ~GroupNormPlugin() {
    platform::dynload::cudnnDestroy(handle_);
    platform::dynload::cudnnDestroyTensorDescriptor(desc_);
    platform::dynload::cudnnDestroyTensorDescriptor(bn_desc_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "group_norm_plugin";
  }

  GroupNormPlugin* clone() const TRT_NOEXCEPT override {
    GroupNormPlugin* ptr = new GroupNormPlugin(epsilon_, groups_);
    ptr->setPluginNamespace(this->getPluginNamespace());
    ptr->input_dims_.assign(this->input_dims_.begin(), this->input_dims_.end());
    ptr->with_fp16_ = this->with_fp16_;
    return ptr;
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

  int initialize() TRT_NOEXCEPT override { return 0; }

  void destroy() noexcept override { delete this; }

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* in_dims,
                                     int nb_inputs) TRT_NOEXCEPT override {
    PADDLE_ENFORCE_EQ(nb_inputs, 3,
                      platform::errors::InvalidArgument(
                          "Invalid number of inputs of group_norm TRT plugin. "
                          "Expected 3, but received %d.",
                          nb_inputs));
    PADDLE_ENFORCE_EQ(
        index, 0,
        platform::errors::InvalidArgument(
            "Index of output should be equal to 0 but received %d.", index));
    return in_dims[0];
  }

  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* input_types,
      int nb_inputs) const TRT_NOEXCEPT override {
    return input_types[0];
  }

  bool isOutputBroadcastAcrossBatch(int output_index,
                                    const bool* input_is_broadcast,
                                    int nb_inputs) const TRT_NOEXCEPT override {
    return false;
  }

  bool canBroadcastInputAcrossBatch(int input_index) const
      TRT_NOEXCEPT override {
    return false;
  }

  void configurePlugin(const nvinfer1::Dims* input_dims, int nb_inputs,
                       const nvinfer1::Dims* output_dims, int nb_outputs,
                       const nvinfer1::DataType* input_types,
                       const nvinfer1::DataType* output_types,
                       const bool* input_is_broadcast,
                       const bool* output_is_broadcast,
                       nvinfer1::PluginFormat float_format,
                       int max_batch_size) TRT_NOEXCEPT override {
    for (int i = 0; i < input_dims->nbDims; i++) {
      input_dims_.push_back(input_dims->d[i]);
    }

    PADDLE_ENFORCE_EQ(input_types[0] == nvinfer1::DataType::kFLOAT ||
                          input_types[0] == nvinfer1::DataType::kHALF,
                      true, platform::errors::InvalidArgument(
                                "The GroupNorm TRT Plugin's input type "
                                "should be float or half."));

    if (input_types[0] == nvinfer1::DataType::kHALF) {
      with_fp16_ = true;
    }

    // const int c = input_dims->d[0];
    // cudaMalloc(&bn_scale_, max_batch_size * c * sizeof(float));
    // cudaMalloc(&bn_bias_, max_batch_size * c * sizeof(float));

    // std::vector<float> ones(c, 1.F);
    // std::vector<float> zeroes(c, 0.F);
    // for (int i = 0; i < max_batch_size; i++) {
    //   cudaMemcpy(bn_scale_ + i * c, ones.data(), sizeof(float) * c,
    //   cudaMemcpyHostToDevice);
    //   cudaMemcpy(bn_bias_ + i * c, zeroes.data(), sizeof(float) * c,
    //   cudaMemcpyHostToDevice);
    // }
  }

#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
#else
  int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return getBaseSerializationSize() + SerializedSize(epsilon_) +
           SerializedSize(groups_) + SerializedSize(input_dims_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
    SerializeValue(&buffer, epsilon_);
    SerializeValue(&buffer, groups_);
    SerializeValue(&buffer, input_dims_);
    SerializeValue(&buffer, with_fp16_);
  }
};

class GroupNormPluginCreator : public nvinfer1::IPluginCreator {
 public:
  GroupNormPluginCreator() {}
  ~GroupNormPluginCreator() override = default;

  void setPluginNamespace(const char* lib_namespace) TRT_NOEXCEPT override {
    namespace_ = std::string(lib_namespace);
  }

  const char* getPluginNamespace() const TRT_NOEXCEPT override {
    return namespace_.c_str();
  }

  const char* getPluginName() const TRT_NOEXCEPT override {
    return "group_norm_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override {
    return &field_collection_;
  }

  nvinfer1::IPluginV2Ext* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc) TRT_NOEXCEPT override {
    return nullptr;
  }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    return new GroupNormPlugin(serial_data, serial_length);
  }

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection field_collection_{0, nullptr};
};

REGISTER_TRT_PLUGIN_V2(GroupNormPluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
