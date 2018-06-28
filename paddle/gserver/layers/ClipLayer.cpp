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

namespace paddle {

/**
 * A layer for clipping the input value by the threshold.
 * \f[
 *   out[i] = \min\left(\max\left(in[i],p_{1}\right),p_{2}\right)
 * \f]
 */

class ClipLayer : public Layer {
 protected:
  double min_;
  double max_;

 public:
  explicit ClipLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(clip, ClipLayer);

bool ClipLayer::init(const LayerMap& layerMap,
                     const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 1U);
  auto layerConf = config_.inputs(0).clip_conf();
  min_ = layerConf.min();
  max_ = layerConf.max();
  CHECK_LT(min_, max_);
  return true;
}

void ClipLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr inV = getInputValue(0);
  resetOutput(inV->getHeight(), inV->getWidth());
  MatrixPtr outV = getOutputValue();
  outV->copyFrom(*inV);
  outV->clip(min_, max_);
}

void ClipLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inV = getInputValue(0);
  MatrixPtr inG = getInputGrad(0);
  if (inG) {
    MatrixPtr outV = getOutputValue();
    MatrixPtr outG = getOutputGrad();
    MatrixPtr tmpMtx;
    Matrix::resizeOrCreate(
        tmpMtx, outG->getHeight(), outG->getWidth(), false, useGpu_);
    tmpMtx->clipDerivative(*inV, min_, max_);
    inG->addDotMul(*outG, *tmpMtx, 1, 1);
  }
}

}  // namespace paddle
