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
using half = phi::dtype::float16;
template <typename T>
class TransformerDecoderPluginDynamic : public DynamicPluginTensorRT {
 public:
  TransformerDecoderPluginDynamic(const T* bias_data, const int bias_size,
                                  const int head_number, const int head_size,
                                  const float scale, const bool with_fp16)
      : head_number_(head_number), head_size_(head_size), scale_(scale), with_fp16_(with_fp16)  {
    bias_.resize(bias_size);
    std::copy(bias_data, bias_data + bias_size, bias_.data());
  }

  TransformerDecoderPluginDynamic(void const* serialData, size_t serialLength);
  ~TransformerDecoderPluginDynamic() {}
  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    auto ptr = new TransformerDecoderPluginDynamic(bias_.data(), bias_.size(), head_number_,
                                      head_size_, scale_, with_fp16_);
    ptr->p_gpu_bias_ = p_gpu_bias_;
    return ptr;
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "TransformerDecoder_plugin_dynamic";
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
  std::vector<T> bias_;
  T* p_gpu_bias_{nullptr};
  int head_number_;
  int head_size_;
  float scale_;
  bool with_fp16_;
};

class TransformerDecoderPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "TransformerDecoder_plugin_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    // FIXME(wanghaoshuang): remove template args 
    return new TransformerDecoderPluginDynamic<half>(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(TransformerDecoderPluginDynamicCreator);


}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
