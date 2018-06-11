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

#include "Layer.h"
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"

namespace paddle {
/**
 * @brief A layer for resizing a minibatch matrix h*w to h'*w'
 * @note
 * origin matrix height * width)
 * resize matrix: (height * width / size) * size
 */
class ResizeLayer : public Layer {
 public:
  explicit ResizeLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback) override;
};

REGISTER_LAYER(resize, ResizeLayer);

bool ResizeLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;
  CHECK_EQ(1U, inputLayers_.size());

  setNeedSequenceInfo(false);
  return true;
}

void ResizeLayer::forward(PassType passType) {
  Layer::forward(passType);
  const Argument& input = getInput(0);
  size_t height = input.value->getHeight();
  size_t width = input.value->getWidth();
  CHECK_EQ((height * width) % getSize(), 0UL);

  reserveOutput(height * width / getSize(), getSize());
  MatrixPtr tmp =
      Matrix::create(output_.value->getData(), height, width, false, useGpu_);
  tmp->assign(*input.value);
}

void ResizeLayer::backward(const UpdateCallback& callback) {
  const Argument& input = getInput(0);
  size_t height = input.value->getHeight();
  size_t width = input.value->getWidth();

  if (!input.grad) {
    return;
  }

  MatrixPtr tmp = Matrix::create(input.grad->getData(),
                                 height * width / getSize(),
                                 getSize(),
                                 false,
                                 useGpu_);
  tmp->add(*output_.grad);
}

}  // namespace paddle
