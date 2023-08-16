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
#include <memory>
#include <vector>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {
class SkipMergeLayernormPluginDynamic : public DynamicPluginTensorRT {
 public:
  SkipMergeLayernormPluginDynamic(const float* bias_d,
                                  const size_t bias_num,
                                  const float* scale_d,
                                  const size_t scale_num,
                                  const float eps,
                                  const int begin_norm_axis,
                                  const bool with_fp16,
                                  std::shared_ptr<void> bias_device = nullptr,
                                  std::shared_ptr<void> scale_device = nullptr);

  SkipMergeLayernormPluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &bias_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &begin_norm_axis_);
    DeserializeValue(&serialData, &serialLength, &with_fp16_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new SkipMergeLayernormPluginDynamic(bias_.data(),
                                               bias_.size(),
                                               scale_.data(),
                                               scale_.size(),
                                               eps_,
                                               begin_norm_axis_,
                                               with_fp16_,
                                               bias_device_,
                                               scale_device_);
  }
  const char* getPluginType() const TRT_NOEXCEPT override {
    return "skip_merge_layernorm_plugin_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override { return 0; }
  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(bias_) + SerializedSize(scale_) +
           SerializedSize(eps_) + SerializedSize(begin_norm_axis_) +
           SerializedSize(with_fp16_);
  }
  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, begin_norm_axis_);
    SerializeValue(&buffer, with_fp16_);
  }
  nvinfer1::DimsExprs getOutputDimensions(
      int output_index,
      const nvinfer1::DimsExprs* inputs,
      int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder)  // NOLINT
      TRT_NOEXCEPT override;
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) TRT_NOEXCEPT override;

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

 private:
  std::vector<float> bias_;
  std::vector<float> scale_;
  float eps_;
  int begin_norm_axis_;
  bool with_fp16_;
  std::shared_ptr<void> bias_device_ = nullptr;
  std::shared_ptr<void> scale_device_ = nullptr;
};
class SkipMergeLayernormPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "skip_merge_layernorm_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new SkipMergeLayernormPluginDynamic(serial_data, serial_length);
  }
};

REGISTER_TRT_PLUGIN_V2(SkipMergeLayernormPluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
