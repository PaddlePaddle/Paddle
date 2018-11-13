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

#include <cassert>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>
#include "NvInfer.h"

#include "paddle/fluid/inference/tensorrt/plugin/serialize.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class PluginTensorRT : public nvinfer1::IPluginExt {
 public:
  PluginTensorRT() {}
  PluginTensorRT(const void* serialized_data, size_t length) {}
  nvinfer1::Dims const& getInputDims(int index) const {
    return input_dims_.at(index);
  }
  size_t getMaxBatchSize() const { return max_batch_size_; }
  nvinfer1::DataType getDataType() const { return data_type_; }
  nvinfer1::PluginFormat getDataFormat() const { return data_format_; }
  virtual const char* getPluginVersion() const { return "1"; }
  size_t getWorkspaceSize(int) const override { return 0; }
  void terminate() override {}
  virtual ~PluginTensorRT() {}

  // The following functions need to be overrided in the subclass.
  virtual nvinfer1::IPluginExt* clone() const = 0;
  virtual const char* getPluginType() const = 0;
  int initialize() override { return 0; }
  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override;
  void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                           const nvinfer1::Dims* outputDims, int nbOutputs,
                           nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int maxBatchSize) override;
  virtual void serialize(void* buffer) = 0;
  virtual size_t getSerializationSize() = 0;

 protected:
  void deserializeBase(void const*& serialData, size_t& serialLength);
  size_t getBaseSerializationSize();
  void serializeBase(void*& buffer);

  std::vector<nvinfer1::Dims> input_dims_;
  size_t max_batch_size_;
  nvinfer1::DataType data_type_;
  nvinfer1::PluginFormat data_format_;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
