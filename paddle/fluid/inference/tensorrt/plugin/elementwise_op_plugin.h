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
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class ElementWisePlugin : public PluginTensorRT {
 public:
  ElementWisePlugin(std::string type, std::vector<int> const &shape_x,
                    std::vector<int> const &shape_y, int axis)
      : type_(type),
        shape_x_(shape_x),
        shape_y_(shape_y),
        axis_(axis),
        prev_size_(1),
        midd_size_(1),
        post_size_(1),
        with_weights_(false) {}

  ElementWisePlugin(std::string type, std::vector<int> const &shape_x,
                    std::vector<int> const &shape_y, int axis,
                    TensorRTEngine::Weight const &weights)
      : type_(type),
        shape_x_(shape_x),
        shape_y_(shape_y),
        axis_(axis),
        prev_size_(1),
        midd_size_(1),
        post_size_(1),
        with_weights_(true),
        weights_(weights) {}

  ElementWisePlugin(void const *serial_data, size_t serial_length) {
    deserializeBase(serial_data, serial_length);
    DeserializeValue(&serial_data, &serial_length, &axis_);
    DeserializeValue(&serial_data, &serial_length, &shape_x_);
    DeserializeValue(&serial_data, &serial_length, &shape_y_);
  }

  ElementWisePlugin *clone() const override {
    // return new ElementWisePlugin(shape_x_, shape_y_, axis_);
    return nullptr;
  }

  const char *getPluginType() const override { return "elementwise"; }

  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims *input_dims,
                                     int num_inputs) override;

  int initialize() override;

  // execute the layer
  int enqueue(int batch_size, const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream);

 protected:
  size_t getSerializationSize() override {
    return SerializedSize(axis_) + SerializedSize(shape_x_) +
           SerializedSize(shape_y_) + getBaseSerializationSize();
  }

  void serialize(void *buffer) override {
    serializeBase(buffer);
    SerializeValue(&buffer, axis_);
    SerializeValue(&buffer, shape_x_);
    SerializeValue(&buffer, shape_y_);
  }

  std::string type_;
  std::vector<int> shape_x_;
  std::vector<int> shape_y_;
  int axis_;
  int prev_size_;
  int midd_size_;
  int post_size_;
  bool with_weights_;
  TensorRTEngine::Weight weights_;
};

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
