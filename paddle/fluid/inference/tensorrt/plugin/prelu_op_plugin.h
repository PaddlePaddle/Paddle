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

#include <string>
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class PReluPlugin : public PluginTensorRT {
  TensorRTEngine::Weight alpha_;
  std::string mode_;

 protected:
  size_t getSerializationSize() override {
    // return getBaseSerializationSize(alpha_) + SerializedSize(mode_);
    return 0;
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void *buffer) override {
    // serializeBase(buffer);
    // SerializeValue(&buffer, alpha_);
    // SerializeValue(&buffer, mode_);
  }

 public:
  PReluPlugin(TensorRTEngine::Weight const &alpha, std::string const &mode)
      : alpha_(alpha), mode_(mode) {}

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  PReluPlugin(void const *serialData, size_t serialLength) {
    // deserializeBase(serialData, serialLength);
    // DeserializeValue(&serialData, &serialLength, &alpha_);
    // DeserializeValue(&serialData, &serialLength, &mode_);
  }

  PReluPlugin *clone() const override { return new PReluPlugin(alpha_, mode_); }

  const char *getPluginType() const override { return "prelu"; }
  int getNbOutputs() const override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs,
                                     int nbInputDims) override;
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
};

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
