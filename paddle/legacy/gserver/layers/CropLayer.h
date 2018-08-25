/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "Layer.h"

namespace paddle {

/**
 * \brief  This layer crop input according to the specify conf.
 *         input_0: input to be cropped
 *         input_1: optional reference input
 *         axis: start dimension to be croped
 *         offset: offset of cropping  in each dimension
 *         shape: if reference input layer was not setted,
 *                  crop input as this shape conf
 */
class CropLayer : public Layer {
 public:
  explicit CropLayer(const LayerConfig& config) : Layer(config) {}

  ~CropLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 protected:
  void setOutDims();
  void setInDims();

  int32_t crop_axis_;
  std::vector<uint32_t> crop_offsets_;
  std::vector<uint32_t> crop_corner_;
  TensorShape inDims_;
  TensorShape targetDims_;
  TensorShape outDims_;
};
}  // namespace paddle
