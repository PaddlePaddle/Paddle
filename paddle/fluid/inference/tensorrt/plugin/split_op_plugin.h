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
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SplitPlugin : public PluginTensorRT {
  int axis_;
  std::vector<int> output_length_;
  int nx_, ny_, nz_;
  std::vector<int> segment_offsets_;

 protected:
  size_t getSerializationSize() override {
    return SerializedSize(axis_) + SerializedSize(output_length_) +
           getBaseSerializationSize();
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void *buffer) override {
    serializeBase(buffer);
    SerializeValue(&buffer, axis_);
    SerializeValue(&buffer, output_length_);
  }

 public:
  SplitPlugin(int axis, std::vector<int> const &output_lengths)
      : axis_(axis), output_length_(output_lengths) {
    assert(axis <= nvinfer1::Dims::MAX_DIMS);
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  SplitPlugin(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &axis_);
    DeserializeValue(&serialData, &serialLength, &output_length_);
  }

  SplitPlugin *clone() const override {
    return new SplitPlugin(axis_, output_length_);
  }

  const char *getPluginType() const override { return "split"; }
  int getNbOutputs() const override { return output_length_.size(); }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs,
                                     int nbInputDims) override;
  int initialize() override;
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
