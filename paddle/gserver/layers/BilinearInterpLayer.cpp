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

#include "BilinearInterpLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(bilinear_interp, BilinearInterpLayer);

size_t BilinearInterpLayer::getSize() {
  inImgH_ = inputLayers_[0]->getOutput().getFrameHeight();
  inImgW_ = inputLayers_[0]->getOutput().getFrameWidth();

  const BilinearInterpConfig& conf = config_.inputs(0).bilinear_interp_conf();
  if (inImgH_ == 0) {
    inImgH_ = conf.image_conf().img_size_y();
  }
  if (inImgW_ == 0) {
    inImgW_ = conf.image_conf().img_size();
  }

  outImgH_ = conf.out_size_y();
  outImgW_ = conf.out_size_x();
  numChannels_ = conf.image_conf().channels();

  CHECK(outImgH_ > 0 && outImgW_ > 0);
  CHECK(inImgH_ > 0 && inImgW_ > 0);
  CHECK(numChannels_);

  ratioH_ =
      (outImgH_ > 1) ? static_cast<real>(inImgH_ - 1) / (outImgH_ - 1) : 0.f;
  ratioW_ =
      (outImgW_ > 1) ? static_cast<real>(inImgW_ - 1) / (outImgW_ - 1) : 0.f;

  getOutput().setFrameHeight(outImgH_);
  getOutput().setFrameWidth(outImgW_);
  return outImgH_ * outImgW_ * numChannels_;
}

bool BilinearInterpLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(1, config_.inputs_size());

  return true;
}

void BilinearInterpLayer::forward(PassType passType) {
  Layer::forward(passType);

  size_t batchSize = getInput(0).getBatchSize();
  size_t size = getSize();
  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    resetOutput(batchSize, size);
  }

  MatrixPtr inV = getInputValue(0);
  MatrixPtr outV = getOutputValue();
  {
    REGISTER_TIMER_INFO("FwBilinearInterpTimer", getName().c_str());
    outV->bilinearForward(*inV,
                          inImgH_,
                          inImgW_,
                          outImgH_,
                          outImgW_,
                          numChannels_,
                          ratioH_,
                          ratioW_);
  }
}

void BilinearInterpLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  MatrixPtr inputG = getInputGrad(0);
  MatrixPtr outG = getOutputGrad();
  {
    REGISTER_TIMER_INFO("BwBilinearInterpTimer", getName().c_str());
    if (inputG) {
      inputG->bilinearBackward(*outG,
                               outImgH_,
                               outImgW_,
                               inImgH_,
                               inImgW_,
                               numChannels_,
                               ratioH_,
                               ratioW_);
    }
  }
}
}  // namespace paddle
