// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class YoloBoxHeadPlugin : public PluginTensorRT {
 public:
  explicit YoloBoxHeadPlugin(const std::vector<int>& anchors,
                             const int class_num)
      : anchors_(anchors), class_num_(class_num) {}

  YoloBoxHeadPlugin(const void* data, size_t length) {
    deserializeBase(data, length);
    DeserializeValue(&data, &length, &anchors_);
    DeserializeValue(&data, &length, &class_num_);
  }

  ~YoloBoxHeadPlugin() override{};

  nvinfer1::IPluginV2* clone() const TRT_NOEXCEPT override {
    return new YoloBoxHeadPlugin(anchors_, class_num_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "yolo_box_head_plugin";
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

  int initialize() TRT_NOEXCEPT override { return 0; }

  nvinfer1::Dims getOutputDimensions(int index,
                                     const nvinfer1::Dims* inputs,
                                     int nb_input_dims) TRT_NOEXCEPT override {
    assert(index == 0);
    assert(nb_input_dims == 1);
    return inputs[0];
  }

  int enqueue(int batch_size,
              const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
              void** outputs,
#else
              void* const* outputs,
#endif
              void* workspace,
              cudaStream_t stream) TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    return getBaseSerializationSize() + SerializedSize(anchors_) +
           SerializedSize(class_num_);
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    serializeBase(buffer);
    SerializeValue(&buffer, anchors_);
    SerializeValue(&buffer, class_num_);
  }

 private:
  std::vector<int> anchors_;
  int class_num_;
  std::string namespace_;
};

class YoloBoxHeadPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "yolo_box_head_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length)
      TRT_NOEXCEPT override {
    return new YoloBoxHeadPlugin(serial_data, serial_length);
  }
};

REGISTER_TRT_PLUGIN_V2(YoloBoxHeadPluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
