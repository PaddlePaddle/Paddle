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

#include <numeric>
#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class Conv2dTransposePlugin : public PluginTensorRT {
  float *weight_data_;
  int weight_data_num_;
  std::vector<int> ksizes_;
  std::vector<int> strides_;
  std::vector<int> paddings_;
  std::vector<int> dilations_;
  int groups_;
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  std::vector<int> filter_shape_;
  int gpu_id_;

 protected:
  size_t getSerializationSize() override {
    // return getBaseSerializationSize(alpha_) + SerializedSize(mode_);
    return 0;
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void *buffer) override {
    // Paddle-trt does not support serialization yet
    // This will be supported later.
  }

 public:
  Conv2dTransposePlugin(float *weight_data, int weight_data_num,
                        std::vector<int> ksizes, std::vector<int> strides,
                        std::vector<int> paddings, std::vector<int> dilations,
                        int groups, std::vector<int> input_shape, int gpu_id)
      : weight_data_(weight_data),
        weight_data_num_(weight_data_num),
        ksizes_(ksizes),
        strides_(strides),
        paddings_(paddings),
        dilations_(dilations),
        groups_(groups),
        input_shape_(input_shape),
        gpu_id_(gpu_id) {
    output_shape_ = input_shape_;
    int output_channel =
        weight_data_num_ / (ksizes[0] * ksizes[1] * input_shape_[0]);
    int filter_extent_h = dilations[0] * (ksizes[0] - 1) + 1;
    int output_h =
        (input_shape[1] - 1) * strides[0] - 2 * paddings[0] + filter_extent_h;

    int filter_extent_w = dilations[1] * (ksizes[1] - 1) + 1;
    int output_w =
        (input_shape[2] - 1) * strides[1] - 2 * paddings[1] + filter_extent_w;

    output_shape_[0] = output_channel;
    output_shape_[1] = output_h;
    output_shape_[2] = output_w;

    filter_shape_.resize(4, 1);
    filter_shape_[0] = input_shape[0];
    filter_shape_[1] = output_channel;
    filter_shape_[2] = ksizes[0];
    filter_shape_[3] = ksizes[1];
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  Conv2dTransposePlugin(void const *serialData, size_t serialLength) {
    // Paddle-trt does not support serialization yet
    // This will be supported later.
  }

  Conv2dTransposePlugin *clone() const override {
    return new Conv2dTransposePlugin(weight_data_, weight_data_num_, ksizes_,
                                     strides_, paddings_, dilations_, groups_,
                                     input_shape_, gpu_id_);
  }

  const char *getPluginType() const override { return "conv2d_transpose"; }
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
