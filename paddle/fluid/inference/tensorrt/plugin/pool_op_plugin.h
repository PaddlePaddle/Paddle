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
#include <stdio.h>
#include <cassert>
#include <string>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class PoolPlugin : public PluginTensorRT {
 protected:
  size_t getSerializationSize() override {
    return SerializedSize(getPluginType()) + SerializedSize(ceil_mode_) +
           SerializedSize(pool_type_) + SerializedSize(adaptive_) +
           SerializedSize(ksize_) + SerializedSize(strides_) +
           SerializedSize(paddings_) + SerializedSize(input_shape_) +
           SerializedSize(output_shape_) + getBaseSerializationSize();
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  void serialize(void *buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, ceil_mode_);
    SerializeValue(&buffer, pool_type_);
    SerializeValue(&buffer, adaptive_);
    SerializeValue(&buffer, ksize_);
    SerializeValue(&buffer, strides_);
    SerializeValue(&buffer, paddings_);
    SerializeValue(&buffer, input_shape_);
    SerializeValue(&buffer, output_shape_);
  }

 public:
  enum class PoolType {
    max = 0,
    avg,
  };
  PoolPlugin() {}
  PoolPlugin(bool ceil_mode, PoolType pool_type, bool adaptive,
             std::vector<int> ksize, std::vector<int> strides,
             std::vector<int> paddings, std::vector<int> input_shape)
      : ceil_mode_(ceil_mode),
        pool_type_(pool_type),
        adaptive_(adaptive),
        ksize_(ksize),
        strides_(strides),
        paddings_(paddings),
        input_shape_(input_shape) {
    output_shape_ = input_shape_;
    if (adaptive_) {
      output_shape_[1] = ksize[0];
      output_shape_[2] = ksize[1];
    } else {
      int output_h, output_w;
      if (!ceil_mode_) {
        output_h =
            (input_shape[1] - ksize_[0] + 2 * paddings_[0]) / strides_[0] + 1;
        output_w =
            (input_shape[2] - ksize_[1] + 2 * paddings_[1]) / strides_[1] + 1;
      } else {
        output_h =
            (input_shape[1] - ksize_[0] + 2 * paddings_[0] + strides_[0] - 1) /
                strides_[0] +
            1;
        output_w =
            (input_shape[2] - ksize_[1] + 2 * paddings_[1] + strides_[1] - 1) /
                strides_[1] +
            1;
      }
      output_shape_[1] = output_h;
      output_shape_[2] = output_w;
    }
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  PoolPlugin(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &ceil_mode_);
    DeserializeValue(&serialData, &serialLength, &pool_type_);
    DeserializeValue(&serialData, &serialLength, &adaptive_);
    DeserializeValue(&serialData, &serialLength, &ksize_);
    DeserializeValue(&serialData, &serialLength, &strides_);
    DeserializeValue(&serialData, &serialLength, &paddings_);
    DeserializeValue(&serialData, &serialLength, &input_shape_);
    DeserializeValue(&serialData, &serialLength, &output_shape_);
  }

  PoolPlugin *clone() const override {
    return new PoolPlugin(ceil_mode_, pool_type_, adaptive_, ksize_, strides_,
                          paddings_, input_shape_);
  }

  const char *getPluginType() const override { return "pool_plugin"; }
  int getNbOutputs() const override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims *inputs,
                                     int nbInputDims) override;
  int initialize() override { return 0; }
  int enqueue(int batchSize, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;

 private:
  bool ceil_mode_;
  PoolType pool_type_;
  bool adaptive_;
  std::vector<int> ksize_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
};

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
