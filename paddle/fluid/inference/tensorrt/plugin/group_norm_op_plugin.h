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
class GroupNormPlugin : public PluginTensorRT {
 public:
  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return getBaseSerializationSize() + SerializedSize(scale_) +
           SerializedSize(bias_) + SerializedSize(eps_) +
           SerializedSize(groups_) + SerializedSize(mean_shape_) +
           SerializedSize(variance_shape_);
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, groups_);
    SerializeValue(&buffer, mean_shape_);
    SerializeValue(&buffer, variance_shape_);
  }

  GroupNormPlugin(const float* scale,
                  const int scale_num,
                  const float* bias,
                  const int bias_num,
                  float eps,
                  int groups,
                  std::vector<int64_t> mean_shape,
                  std::vector<int64_t> variance_shape)
      : groups_(groups),
        eps_(eps),
        mean_shape_(mean_shape),
        variance_shape_(variance_shape) {
    scale_.resize(scale_num);
    bias_.resize(bias_num);
    std::copy(scale, scale + scale_num, scale_.data());
    std::copy(bias, bias + bias_num, bias_.data());
  }
  GroupNormPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &bias_);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &groups_);
    DeserializeValue(&serialData, &serialLength, &mean_shape_);
    DeserializeValue(&serialData, &serialLength, &variance_shape_);
  }
  ~GroupNormPlugin() {}
  int initialize() TRT_NOEXCEPT override;
  GroupNormPlugin* clone() const TRT_NOEXCEPT override {
    return new GroupNormPlugin(scale_.data(),
                               scale_.size(),
                               bias_.data(),
                               bias_.size(),
                               eps_,
                               groups_,
                               mean_shape_,
                               variance_shape_);
  }
  const char* getPluginType() const TRT_NOEXCEPT override {
    return "groupnorm_plugin";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* inputs,
                                     int nbInputDims) TRT_NOEXCEPT override;

#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize,
              const void* const* inputs,
              void** outputs,
#else
  int enqueue(int batchSize,
              const void* const* inputs,
              void* const* outputs,
#endif
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;

 private:
  std::vector<float> scale_;
  std::vector<float> bias_;
  framework::Tensor scale_t;
  framework::Tensor bias_t;
  framework::Tensor mean_t;
  framework::Tensor variance_t;
  int groups_;
  float eps_;
  std::vector<int64_t> mean_shape_;
  std::vector<int64_t> variance_shape_;
};
class GroupNormPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "groupnorm_plugin";
  }
  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new GroupNormPlugin(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(GroupNormPluginCreator);

class GroupNormPluginDynamic : public DynamicPluginTensorRT {
 public:
  GroupNormPluginDynamic(const float* scale,
                         const int scale_num,
                         const float* bias,
                         const int bias_num,
                         float eps,
                         int groups,
                         std::vector<int64_t> mean_shape,
                         std::vector<int64_t> variance_shape)
      : groups_(groups),
        eps_(eps),
        mean_shape_(mean_shape),
        variance_shape_(variance_shape) {
    scale_.resize(scale_num);
    bias_.resize(bias_num);
    std::copy(scale, scale + scale_num, scale_.data());
    std::copy(bias, bias + bias_num, bias_.data());
  }

  GroupNormPluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &bias_);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &groups_);
    DeserializeValue(&serialData, &serialLength, &mean_shape_);
    DeserializeValue(&serialData, &serialLength, &variance_shape_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new GroupNormPluginDynamic(scale_.data(),
                                      scale_.size(),
                                      bias_.data(),
                                      bias_.size(),
                                      eps_,
                                      groups_,
                                      mean_shape_,
                                      variance_shape_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "groupnorm_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override { return 0; }

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(scale_) + SerializedSize(bias_) +
           SerializedSize(eps_) + SerializedSize(groups_) +
           SerializedSize(mean_shape_) + SerializedSize(variance_shape_);
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, groups_);
    SerializeValue(&buffer, mean_shape_);
    SerializeValue(&buffer, variance_shape_);
  }
  nvinfer1::DimsExprs getOutputDimensions(int output_index,
                                          const nvinfer1::DimsExprs* inputs,
                                          int nb_inputs,
                                          nvinfer1::IExprBuilder& expr_builder)
      TRT_NOEXCEPT override;

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
              const void* const* inputs,
              void* const* outputs,
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const
      TRT_NOEXCEPT override;

  void destroy() TRT_NOEXCEPT override { delete this; }
  // void terminate() TRT_NOEXCEPT override;

 private:
  std::vector<float> scale_;
  std::vector<float> bias_;
  framework::Tensor scale_t;
  framework::Tensor bias_t;
  framework::Tensor mean_t;
  framework::Tensor variance_t;
  int groups_;
  float eps_;
  std::vector<int64_t> mean_shape_;
  std::vector<int64_t> variance_shape_;
};
class GroupNormPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "groupnorm_plugin_dynamic";
  }
  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new GroupNormPluginDynamic(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(GroupNormPluginDynamicCreator);
}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
