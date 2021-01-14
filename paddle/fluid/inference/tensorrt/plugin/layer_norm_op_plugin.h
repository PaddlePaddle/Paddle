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

class LayerNormPlugin : public PluginTensorRT {
  std::vector<float> bias_;
  std::vector<float> scale_;
  framework::Tensor scale_t;
  framework::Tensor bias_t;
  framework::Tensor mean_t;
  framework::Tensor variance_t;
  int begin_norm_axis_;
  float eps_;
  std::vector<int64_t> mean_shape_;
  std::vector<int64_t> variance_shape_;

 protected:
  size_t getSerializationSize() override {
    return getBaseSerializationSize() + SerializedSize(bias_) +
           SerializedSize(scale_) + SerializedSize(begin_norm_axis_) +
           SerializedSize(eps_) + SerializedSize(mean_shape_) +
           SerializedSize(variance_shape_) + SerializedSize(getPluginType());
  }

  // TRT will call this func when we need to serialize the configuration of
  // tensorrt.
  // It should not be called by users.
  void serialize(void *buffer) override {
    SerializeValue(&buffer, getPluginType());
    serializeBase(buffer);
    SerializeValue(&buffer, bias_);
    SerializeValue(&buffer, scale_);
    SerializeValue(&buffer, begin_norm_axis_);
    SerializeValue(&buffer, eps_);
    SerializeValue(&buffer, mean_shape_);
    SerializeValue(&buffer, variance_shape_);
  }

 public:
  LayerNormPlugin(const float *bias, const int bias_num, const float *scale,
                  const int scale_num, int begin_norm_axis, float eps,
                  std::vector<int64_t> mean_shape,
                  std::vector<int64_t> variance_shape)
      : begin_norm_axis_(begin_norm_axis),
        eps_(eps),
        mean_shape_(mean_shape),
        variance_shape_(variance_shape) {
    bias_.resize(bias_num);
    scale_.resize(scale_num);
    std::copy(bias, bias + bias_num, bias_.data());
    std::copy(scale, scale + scale_num, scale_.data());
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  LayerNormPlugin(void const *serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &bias_);
    DeserializeValue(&serialData, &serialLength, &scale_);
    DeserializeValue(&serialData, &serialLength, &begin_norm_axis_);
    DeserializeValue(&serialData, &serialLength, &eps_);
    DeserializeValue(&serialData, &serialLength, &mean_shape_);
    DeserializeValue(&serialData, &serialLength, &variance_shape_);
  }
  ~LayerNormPlugin() {}
  int initialize() override;

  LayerNormPlugin *clone() const override {
    return new LayerNormPlugin(bias_.data(), bias_.size(), scale_.data(),
                               scale_.size(), begin_norm_axis_, eps_,
                               mean_shape_, variance_shape_);
  }

  const char *getPluginType() const override { return "layer_norm_plugin"; }
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
