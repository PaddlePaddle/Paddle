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
                             const int class_num, const float conf_thresh,
                             const int downsample_ratio, const bool clip_bbox,
                             const float scale_x_y)
      : anchors_(anchors),
        class_num_(class_num),
        conf_thresh_(conf_thresh),
        downsample_ratio_(downsample_ratio),
        clip_bbox_(clip_bbox),
        scale_x_y_(scale_x_y) {}

  YoloBoxHeadPlugin(const void* data, size_t length) {
    DeserializeValue(&data, &length, &anchors_);
    DeserializeValue(&data, &length, &class_num_);
    DeserializeValue(&data, &length, &conf_thresh_);
    DeserializeValue(&data, &length, &downsample_ratio_);
    DeserializeValue(&data, &length, &clip_bbox_);
    DeserializeValue(&data, &length, &scale_x_y_);
  }

  ~YoloBoxHeadPlugin() override{};

  nvinfer1::IPluginV2* clone() const TRT_NOEXCEPT override {
    return new YoloBoxHeadPlugin(anchors_, class_num_, conf_thresh_,
                                 downsample_ratio_, clip_bbox_, scale_x_y_);
  }

  const char* getPluginType() const TRT_NOEXCEPT override {
    return "yolo_box_head_plugin";
  }

  int getNbOutputs() const TRT_NOEXCEPT override { return 1; }

  int initialize() TRT_NOEXCEPT override { return 0; }

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nb_input_dims) TRT_NOEXCEPT override {
    assert(index == 0);
    assert(nb_input_dims == 1);
    return inputs[0];
  }

  int enqueue(int batch_size, const void* const* inputs,
#if IS_TRT_VERSION_LT(8000)
              void** outputs,
#else
              void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override {
    size_t serialize_size = 0;
    serialize_size += SerializedSize(anchors_);
    serialize_size += SerializedSize(class_num_);
    serialize_size += SerializedSize(conf_thresh_);
    serialize_size += SerializedSize(downsample_ratio_);
    serialize_size += SerializedSize(clip_bbox_);
    serialize_size += SerializedSize(scale_x_y_);
    return serialize_size;
  }

  void serialize(void* buffer) const TRT_NOEXCEPT override {
    SerializeValue(&buffer, anchors_);
    SerializeValue(&buffer, class_num_);
    SerializeValue(&buffer, conf_thresh_);
    SerializeValue(&buffer, downsample_ratio_);
    SerializeValue(&buffer, clip_bbox_);
    SerializeValue(&buffer, scale_x_y_);
  }

 private:
  std::vector<int> anchors_;
  int* anchors_device_;
  int class_num_;
  float conf_thresh_;
  int downsample_ratio_;
  bool clip_bbox_;
  float scale_x_y_;
  std::string namespace_;
};

class YoloBoxHeadPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const TRT_NOEXCEPT override {
    return "hard_swish_plugin";
  }

  const char* getPluginVersion() const TRT_NOEXCEPT override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serial_data,
      size_t serial_length) TRT_NOEXCEPT override {
    return new YoloBoxHeadPlugin(serial_data, serial_length);
  }
};

REGISTER_TRT_PLUGIN_V2(YoloBoxHeadPluginCreator);

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
