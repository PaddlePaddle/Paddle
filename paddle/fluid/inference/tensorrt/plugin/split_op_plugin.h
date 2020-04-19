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

#include <thrust/device_vector.h>
#include <utility>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class SplitPlugin : public PluginTensorRT {
 public:
  SplitPlugin() {}
  SplitPlugin(int axis, std::vector<int> const& output_lengths)
      : axis_(axis), same_shape_(true), output_length_(output_lengths) {}

  SplitPlugin(void const* serial_data, size_t serial_length) {
    deserializeBase(serial_data, serial_length);
    DeserializeValue(&serial_data, &serial_length, &axis_);
    DeserializeValue(&serial_data, &serial_length, &output_length_);
  }

  SplitPlugin* clone() const override {
    return new SplitPlugin(axis_, output_length_);
  }

  const char* getPluginType() const override { return "split_plugin"; }
  int getNbOutputs() const override { return output_length_.size(); }
  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* input_dims,
                                     int num_inputs) override;

  int initialize() override;
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
              void* workspace, cudaStream_t stream) override;

 protected:
  size_t getSerializationSize() override {
    return SerializedSize(getPluginType()) + SerializedSize(axis_) +
           SerializedSize(output_length_) + getBaseSerializationSize();
  }

  void serialize(void* buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, axis_);
    SerializeValue(&buffer, output_length_);
  }

  int axis_;
  int outer_rows_;
  int inner_cols_;
  int axis_shape_;
  bool same_shape_;
  std::vector<int> output_length_;
  std::vector<int> segment_offsets_;
  thrust::device_vector<int> d_segment_offsets_;
  thrust::device_vector<float*> d_output_ptrs_;
};

#if IS_TRT_VERSION_GE(6000)
class SplitPluginDynamic : public DynamicPluginTensorRT {
 public:
  SplitPluginDynamic(int axis, std::vector<int> const& output_lengths)
      : axis_(axis), output_length_(output_lengths) {}

  SplitPluginDynamic(void const* serial_data, size_t serial_length) {}

  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new SplitPluginDynamic(axis_, output_length_);
  }

  const char* getPluginType() const override { return "split_plugin"; }
  int getNbOutputs() const override { return output_length_.size(); }
  int initialize() override;

  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;

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
  std::vector<int> output_length_;
};
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
