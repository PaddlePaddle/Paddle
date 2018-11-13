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
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class SplitPlugin : public PluginTensorRT {
  int axis_;
  std::vector<int> output_length_;
  int nx_, ny_, nz_;
  thrust::device_vector<int> d_segment_offsets_;
  std::vector<int> segment_offsets_;

 protected:
  virtual size_t getSerializationSize() override {
    return serialized_size(axis_) + serialized_size(output_length_) +
           getBaseSerializationSize();
  }

  virtual void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, axis_);
    serialize_value(&buffer, output_length_);
  }

 public:
  SplitPlugin(int axis, std::vector<int> const &output_lengths)
      : axis_(axis), output_length_(output_lengths) {
    assert(axis <= nvinfer1::Dims::MAX_DIMS);
  }

  SplitPlugin(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &axis_);
    deserialize_value(&serialData, &serialLength, &output_length_);
  }

  SplitPlugin *clone() const override {
    return new SplitPlugin(axis_, output_length_);
  }

  virtual const char *getPluginType() const override { return "split"; }
  virtual int getNbOutputs() const override { return output_length_.size(); }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs,
                                             int nbInputDims) override;
  virtual int initialize() override;
  virtual int enqueue(int batchSize, const void *const *inputs, void **outputs,
                      void *workspace, cudaStream_t stream) override;

  void setAxis(int axis) { axis_ = axis; }

  void setOutputLengths(const std::vector<int> &output_lengths) {
    output_length_ = output_lengths;
  }
};

}  // tensorrt
}  // inference
}  // paddle
