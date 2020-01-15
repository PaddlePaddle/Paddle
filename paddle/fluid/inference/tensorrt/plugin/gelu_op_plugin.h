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
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class GeluPlugin : public PluginTensorRT {
 protected:
  size_t getSerializationSize() override {
    return getBaseSerializationSize() + SerializedSize(getPluginType()) +
           SerializedSize(input_volume_);
  }

  // TRT will call this func  to serialize the configuration of TRT
  // It should not be called by users.
  void serialize(void *buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, input_volume_);
  }

 public:
  explicit GeluPlugin(size_t input_volume) : input_volume_(input_volume) {}

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  GeluPlugin(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &input_volume_);
  }

  ~GeluPlugin() {}

  int initialize() override { return 0; }

  GeluPlugin *clone() const override { return new GeluPlugin(input_volume_); }

  const char *getPluginType() const override { return "gelu_plugin"; }
  int getNbOutputs() const override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs,
                                     int nbInputDims) override;
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;

 private:
  size_t input_volume_;
};

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
