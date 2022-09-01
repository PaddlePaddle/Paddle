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

class LayernormShiftPartitionPluginDynamic : public DynamicPluginTensorRT {
 public:
  LayernormShiftPartitionPluginDynamic(
      const float* gamma,
      const float* beta,
      const int param_num,
      int shift_size,
      int window_size,
      int input_resolution,
      float eps,
      bool with_fp16,
      std::shared_ptr<void> gamma_dev = nullptr,
      std::shared_ptr<void> beta_dev = nullptr);

  LayernormShiftPartitionPluginDynamic(void const* serialData,
                                       size_t serialLength);

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override {
    return new LayernormShiftPartitionPluginDynamic(gamma_.data(),
                                                    beta_.data(),
                                                    beta_.size(),
                                                    shift_size_,
                                                    window_size_,
                                                    input_resolution_,
                                                    eps_,
                                                    with_fp16_,
                                                    gamma_dev_,
                                                    beta_dev_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "layernorm_shift_partition_dynamic";
  }
  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }
  int initialize() TRT_NOEXCEPT override { return 0; }

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return SerializedSize(beta_) + SerializedSize(gamma_) +
           SerializedSize(param_num_) + SerializedSize(with_fp16_) +
           SerializedSize(shift_size_) + SerializedSize(window_size_) +
           SerializedSize(input_resolution_) + SerializedSize(eps_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, beta_);
    SerializeValue(&buffer, gamma_);
    SerializeValue(&buffer, param_num_);
    SerializeValue(&buffer, with_fp16_);
    SerializeValue(&buffer, shift_size_);
    SerializeValue(&buffer, window_size_);
    SerializeValue(&buffer, input_resolution_);
    SerializeValue(&buffer, eps_);
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
  bool with_fp16_;
  std::vector<float> gamma_;
  std::vector<float> beta_;
  int window_size_;
  int shift_size_;
  int input_resolution_;
  int param_num_;
  float eps_;
  std::shared_ptr<void> gamma_dev_;
  std::shared_ptr<void> beta_dev_;
};

class LayernormShiftPartitionPluginDynamicCreator
    : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "layernorm_shift_partition_dynamic";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new LayernormShiftPartitionPluginDynamic(serial_data, serial_length);
  }
};

REGISTER_TRT_PLUGIN_V2(LayernormShiftPartitionPluginDynamicCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
