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

  std::vector<float> ones_for_serialize_;
  std::vector<float> zeroes_for_serialize_;

  float* scale_;
  float* bias_;
  std::vector<float> scale_v_;
  std::vector<float> bias_v_;
  std::vector<float> scale_for_serialize_;
  std::vector<float> bias_for_serialize_;
  float* bn_scale_;
  float* bn_bias_;

  bool with_fp16_;

  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t desc_, bn_desc_;

 public:
  explicit GroupNormPlugin(const float epsilon, const int groups,
                           const std::vector<float> scale,
                           const std::vector<float> bias, const bool with_fp16)
      : epsilon_(epsilon),
        groups_(groups),
        scale_v_(scale),
        bias_v_(bias),
        with_fp16_(with_fp16) {
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

    DeserializeValue(&serialData, &serialLength, &ones_for_serialize_);
    DeserializeValue(&serialData, &serialLength, &zeroes_for_serialize_);
    DeserializeValue(&serialData, &serialLength, &scale_for_serialize_);
    DeserializeValue(&serialData, &serialLength, &bias_for_serialize_);
    deserializeToDevice();

    platform::dynload::cudnnCreate(&handle_);
    platform::dynload::cudnnCreateTensorDescriptor(&desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&bn_desc_);
  }

  void deserializeToDevice() {
    const int size = static_cast<int>(ones_for_serialize_.size());
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&bn_scale_, sizeof(float) * size));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&bn_bias_, sizeof(float) * size));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(bn_scale_, ones_for_serialize_.data(),
                                          sizeof(float) * size,
                                          cudaMemcpyHostToDevice));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMemcpy(bn_bias_, zeroes_for_serialize_.data(), sizeof(float) * size,
                   cudaMemcpyHostToDevice));

    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&scale_, sizeof(float) * size));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMalloc(&bias_, sizeof(float) * size));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(scale_, scale_for_serialize_.data(),
                                          sizeof(float) * size,
                                          cudaMemcpyHostToDevice));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(bias_, bias_for_serialize_.data(),
                                          sizeof(float) * size,
                                          cudaMemcpyHostToDevice));
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
    GroupNormPlugin* ptr =
        new GroupNormPlugin(epsilon_, groups_, scale_v_, bias_v_, with_fp16_);
    ptr->setPluginNamespace(this->getPluginNamespace());
    ptr->input_dims_.assign(this->input_dims_.begin(), this->input_dims_.end());
    ptr->scale_ = this->scale_;
    ptr->bias_ = this->bias_;
    ptr->bn_scale_ = this->bn_scale_;
    ptr->bn_bias_ = this->bn_bias_;
    ptr->ones_for_serialize_ = this->ones_for_serialize_;
    ptr->zeroes_for_serialize_ = this->zeroes_for_serialize_;
    ptr->scale_for_serialize_ = this->scale_for_serialize_;
    ptr->bias_for_serialize_ = this->bias_for_serialize_;
    return ptr;
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

  int initialize() TRT_NOEXCEPT override { return 0; }

  void destroy() noexcept override { delete this; }

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* in_dims,
                                     int nb_inputs) TRT_NOEXCEPT override {
    PADDLE_ENFORCE_EQ(nb_inputs, 1,
                      platform::errors::InvalidArgument(
                          "Invalid number of inputs of group_norm TRT plugin. "
                          "Expected 1, but received %d.",
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
    PADDLE_ENFORCE_EQ(input_types[0] == nvinfer1::DataType::kFLOAT ||
                          input_types[0] == nvinfer1::DataType::kHALF,
                      true, platform::errors::InvalidArgument(
                                "The GroupNorm TRT Plugin's input type "
                                "should be float32 or float16."));
    return input_types[0];
  }

  bool supportsFormat(nvinfer1::DataType type, nvinfer1::TensorFormat format)
      const TRT_NOEXCEPT override {
    if (with_fp16_) {
#ifdef TRT_PLUGIN_FP16_AVALIABLE
      return (type == nvinfer1::DataType::kHALF) &&
             (format == nvinfer1::TensorFormat::kLINEAR);
#else
      return (type == nvinfer1::DataType::kFLOAT) &&
             (format == nvinfer1::TensorFormat::kLINEAR);
#endif
    } else {
      return (type == nvinfer1::DataType::kFLOAT) &&
             (format == nvinfer1::TensorFormat::kLINEAR);
    }
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

    const int c = input_dims->d[0];
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMalloc(&bn_scale_, max_batch_size * c * sizeof(float)));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMalloc(&bn_bias_, max_batch_size * c * sizeof(float)));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMalloc(&scale_, max_batch_size * c * sizeof(float)));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaMalloc(&bias_, max_batch_size * c * sizeof(float)));

    std::vector<float> ones(c, 1.F);
    std::vector<float> zeroes(c, 0.F);
    for (int i = 0; i < max_batch_size; i++) {
      ones_for_serialize_.insert(ones_for_serialize_.end(), ones.begin(),
                                 ones.end());
      zeroes_for_serialize_.insert(zeroes_for_serialize_.end(), zeroes.begin(),
                                   zeroes.end());
      scale_for_serialize_.insert(scale_for_serialize_.end(), scale_v_.begin(),
                                  scale_v_.end());
      bias_for_serialize_.insert(bias_for_serialize_.end(), bias_v_.begin(),
                                 bias_v_.end());
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(bn_scale_ + i * c, ones.data(),
                                            sizeof(float) * c,
                                            cudaMemcpyHostToDevice));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(bn_bias_ + i * c, zeroes.data(),
                                            sizeof(float) * c,
                                            cudaMemcpyHostToDevice));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(scale_ + i * c, scale_v_.data(),
                                            sizeof(float) * c,
                                            cudaMemcpyHostToDevice));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(bias_ + i * c, bias_v_.data(),
                                            sizeof(float) * c,
                                            cudaMemcpyHostToDevice));
    }
  }

#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
#else
  int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

  template <typename T>
#if IS_TRT_VERSION_LT(8000)
  int enqueueImpl(const void* const* inputs, void** outputs,
#else
  int enqueueImpl(const void* const* inputs, void* const* outputs,
#endif
                  const cudnnHandle_t& handle,
                  const cudnnTensorDescriptor_t& desc,
                  const cudnnTensorDescriptor_t& bn_desc, float* bn_scale,
                  float* bn_bias, const float eps, const int channel_volume,
                  const int batch_size, const int c,
                  const cudaStream_t& stream);

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return getBaseSerializationSize() + SerializedSize(epsilon_) +
           SerializedSize(groups_) + SerializedSize(input_dims_) +
           SerializedSize(with_fp16_) + SerializedSize(ones_for_serialize_) +
           SerializedSize(zeroes_for_serialize_) +
           SerializedSize(scale_for_serialize_) +
           SerializedSize(bias_for_serialize_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
    SerializeValue(&buffer, epsilon_);
    SerializeValue(&buffer, groups_);
    SerializeValue(&buffer, input_dims_);
    SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, ones_for_serialize_);
    SerializeValue(&buffer, zeroes_for_serialize_);
    SerializeValue(&buffer, scale_for_serialize_);
    SerializeValue(&buffer, bias_for_serialize_);
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
