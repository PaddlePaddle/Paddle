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

class PReluPlugin : public PluginTensorRT {
  std::vector<float> weight_;
  float *p_gpu_weight_;
  std::string mode_;

 protected:
  size_t getSerializationSize() override {
    return getBaseSerializationSize() + SerializedSize(mode_.c_str()) +
           SerializedSize(weight_) + SerializedSize(getPluginType());
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void *buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, weight_);
    SerializeValue(&buffer, mode_.c_str());
  }

 public:
  PReluPlugin(const float *weight, const int weight_num,
              std::string const &mode)
      : mode_(mode) {
    weight_.resize(weight_num);
    std::copy(weight, weight + weight_num, weight_.data());
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  PReluPlugin(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &weight_);
    const char *prelu_mode;
    DeserializeValue(&serialData, &serialLength, &prelu_mode);
    mode_ = std::string(prelu_mode);
  }
  ~PReluPlugin() { cudaFree(p_gpu_weight_); }
  int initialize() override;

  PReluPlugin *clone() const override {
    return new PReluPlugin(weight_.data(), weight_.size(), mode_);
  }

  const char *getPluginType() const override { return "prelu_plugin"; }
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
