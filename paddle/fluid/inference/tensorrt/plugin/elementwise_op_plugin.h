/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class ElementWisePlugin : public PluginTensorRT {
 public:
  ElementWisePlugin(std::string type, nvinfer1::Dims const &dims_x,
                    nvinfer1::Dims const &dims_y, int axis)
      : type_(type),
        dims_x_(dims_x),
        dims_y_(dims_y),
        axis_(axis),
        prev_size_(1),
        midd_size_(1),
        post_size_(1) {}

  ElementWisePlugin(void const *serial_data, size_t serial_length) {
    deserializeBase(serial_data, serial_length);
    const char *elementwise_type;
    DeserializeValue(&serial_data, &serial_length, &elementwise_type);
    type_ = std::string(elementwise_type);
    DeserializeValue(&serial_data, &serial_length, &axis_);
    DeserializeValue(&serial_data, &serial_length, &dims_x_);
    DeserializeValue(&serial_data, &serial_length, &dims_y_);
  }

  ElementWisePlugin *clone() const override {
    // return new ElementWisePlugin(dims_x_, dims_y_, axis_);
    return nullptr;
  }

  const char *getPluginType() const override { return "elementwise_plugin"; }

  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims *input_dims,
                                     int num_inputs) override;

  int initialize() override;

  // execute the layer
  int enqueue(int batch_size, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream);

 protected:
  size_t getSerializationSize() override {
    return SerializedSize(getPluginType()) + SerializedSize(axis_) +
           SerializedSize(dims_x_) + SerializedSize(dims_y_) +
           getBaseSerializationSize();
  }

  void serialize(void *buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, type_.c_str());
    SerializeValue(&buffer, axis_);
    SerializeValue(&buffer, dims_x_);
    SerializeValue(&buffer, dims_y_);
  }

  std::string type_;
  nvinfer1::Dims dims_x_;
  nvinfer1::Dims dims_y_;
  int axis_;
  int prev_size_;
  int midd_size_;
  int post_size_;
};

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
