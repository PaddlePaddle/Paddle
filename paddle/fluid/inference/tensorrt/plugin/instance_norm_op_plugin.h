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
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class InstanceNormPlugin : public PluginTensorRT {
 private:
  float eps_;
  std::vector<float> scale_;
  std::vector<float> bias_;

  phi::DenseTensor scale_t;
  phi::DenseTensor bias_t;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t x_desc_, y_desc_, b_desc_;

 public:
  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return getBaseSerializationSize() + SerializedSize(eps_) +
           SerializedSize(scale_) + SerializedSize(bias_);
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void *buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, bias_);
  }

  explicit InstanceNormPlugin(const float eps,
                              const std::vector<float> scale,
                              const std::vector<float> bias)
      : eps_(eps), scale_(scale), bias_(bias) {
    PADDLE_ENFORCE_EQ(scale.size(),
                      bias.size(),
                      platform::errors::InvalidArgument(
                          "The instanceNorm's scale and bias should be the "
                          "same size. Got scale size = %d, but bias size = %d",
                          scale.size(),
                          bias.size()));
    platform::dynload::cudnnCreate(&handle_);
    platform::dynload::cudnnCreateTensorDescriptor(&x_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&y_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&b_desc_);
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  InstanceNormPlugin(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &bias_);

    platform::dynload::cudnnCreate(&handle_);
    platform::dynload::cudnnCreateTensorDescriptor(&x_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&y_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&b_desc_);
  }

  ~InstanceNormPlugin() {
    platform::dynload::cudnnDestroy(handle_);
    platform::dynload::cudnnDestroyTensorDescriptor(x_desc_);
    platform::dynload::cudnnDestroyTensorDescriptor(y_desc_);
    platform::dynload::cudnnDestroyTensorDescriptor(b_desc_);
  }

  int initialize() TRT_NOEXCEPT override;

  InstanceNormPlugin *clone() const TRT_NOEXCEPT override {
    return new InstanceNormPlugin(eps_, scale_, bias_);
  }

  const char *getPluginType() const TRT_NOEXCEPT override {
    return "instance_norm";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims *inputs,
                                     int nbInputDims) TRT_NOEXCEPT override;

#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize,
              const void *const *inputs,
              void **outputs,
#else
  int enqueue(int batchSize,
              const void *const *inputs,
              void *const *outputs,
#endif
              void *workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;

  bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format)
      const TRT_NOEXCEPT override;
};

class InstanceNormPluginCreator : public TensorRTPluginCreator {
 public:
  const char *getPluginName() const TRT_NOEXCEPT override {
    return "instance_norm";
  }

  const char *getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2 *deserializePlugin(const char *name,
                                         const void *serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new InstanceNormPlugin(serial_data, serial_length);
  }
};

class InstanceNormPluginDynamic : public DynamicPluginTensorRT {
 private:
  float eps_;
  std::vector<float> scale_;
  std::vector<float> bias_;

  phi::DenseTensor scale_t;
  phi::DenseTensor bias_t;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t x_desc_, y_desc_, b_desc_;

 public:
  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(eps_) + SerializedSize(scale_) +
           SerializedSize(bias_);
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void *buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, bias_);
  }

  explicit InstanceNormPluginDynamic(const float eps,
                                     const std::vector<float> scale,
                                     const std::vector<float> bias)
      : eps_(eps), scale_(scale), bias_(bias) {
    PADDLE_ENFORCE_EQ(scale.size(),
                      bias.size(),
                      platform::errors::InvalidArgument(
                          "The instanceNorm's scale and bias should be the "
                          "same size. Got scale size = %d, but bias size = %d",
                          scale.size(),
                          bias.size()));
    platform::dynload::cudnnCreate(&handle_);
    platform::dynload::cudnnCreateTensorDescriptor(&x_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&y_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&b_desc_);
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  InstanceNormPluginDynamic(void const *serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &bias_);

    platform::dynload::cudnnCreate(&handle_);
    platform::dynload::cudnnCreateTensorDescriptor(&x_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&y_desc_);
    platform::dynload::cudnnCreateTensorDescriptor(&b_desc_);
  }

  ~InstanceNormPluginDynamic() {
    platform::dynload::cudnnDestroy(handle_);
    platform::dynload::cudnnDestroyTensorDescriptor(x_desc_);
    platform::dynload::cudnnDestroyTensorDescriptor(y_desc_);
    platform::dynload::cudnnDestroyTensorDescriptor(b_desc_);
  }

  int initialize() TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt *clone() const TRT_NOEXCEPT override {
    return new InstanceNormPluginDynamic(eps_, scale_, bias_);
  }

  const char *getPluginType() const TRT_NOEXCEPT override {
    return "instance_norm_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs *inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder &expr_builder)  // NOLINT
      TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc *inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out,
                       int nbOutputs) TRT_NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const TRT_NOEXCEPT override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs,
              void *const *outputs,
              void *workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }
};

class InstanceNormPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char *getPluginName() const TRT_NOEXCEPT override {
    return "instance_norm_dynamic";
  }

  const char *getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2 *deserializePlugin(const char *name,
                                         const void *serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new InstanceNormPluginDynamic(serial_data, serial_length);
  }
};

REGISTER_TRT_PLUGIN_V2(InstanceNormPluginCreator);
REGISTER_TRT_PLUGIN_V2(InstanceNormPluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
