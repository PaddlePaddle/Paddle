/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "HierarchicalSigmoidLayer.h"
#include "paddle/utils/Util.h"

namespace paddle {

REGISTER_LAYER(hsigmoid, HierarchicalSigmoidLayer);

bool HierarchicalSigmoidLayer::init(const LayerMap& layerMap,
                                    const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK(config_.has_num_classes()) << "num_classes must be specifed in config";
  numClasses_ = config_.num_classes();
  CHECK_GE(numClasses_, (size_t)2);
  codeLength_ = findLastSet(numClasses_ - 1);

  size_t height = numClasses_ - 1;

  /* initialize the weightList */
  // The last input layer is for label
  CHECK(!parameters_.back());
  for (size_t i = 0; i < inputLayers_.size() - 1; i++) {
    size_t width = inputLayers_[i]->getSize();
    // create a new weight
    CHECK_EQ(parameters_[i]->getSize(), width * height);
    Weight* w = new Weight(height, width, parameters_[i]);

    // append the new weight to the list
    weights_.emplace_back(w);
  }

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    CHECK_EQ(biasParameter_->getSize(), numClasses_ - 1);
    biases_.reset(new Weight(1, numClasses_ - 1, biasParameter_));
  }

  return true;
}

void HierarchicalSigmoidLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInputValue(0)->getHeight();
  int size = getSize();
  reserveOutput(batchSize, size);
  Matrix::resizeOrCreate(preOutput_.value,
                         batchSize,
                         codeLength_,
                         /* trans */ false,
                         useGpu(deviceId_));
  Matrix::resizeOrCreate(preOutput_.grad,
                         batchSize,
                         codeLength_,
                         /* trans */ false,
                         useGpu(deviceId_));

  IVectorPtr label = getInput(*getLabelLayer()).ids;

  preOutput_.value->zeroMem();

  /* add the bias-vector */
  if (biases_.get() != NULL) {
    preOutput_.value->addByBitCode(numClasses_, *label, *biases_->getW());
  }
  for (size_t i = 0; i < inputLayers_.size() - 1; ++i) {
    MatrixPtr input = getInputValue(i);
    preOutput_.value->mulByBitCode(
        numClasses_, *label, *weights_[i]->getW(), *input);
  }
  // keep consistent with the clipping in the following softrelu
  preOutput_.value->clip(-40.0, 40.0);
  preOutput_.value->sumByBitCode(numClasses_,
                                 *label,
                                 *output_.value,
                                 -1);  // scaleSum
  preOutput_.value->softrelu(*preOutput_.value);
  MatrixPtr sum =
      Matrix::create(batchSize, 1, /* trans= */ false, useGpu(deviceId_));
  preOutput_.value->rowSum(*sum);
  output_.value->add(*sum);
}

void HierarchicalSigmoidLayer::backward(const UpdateCallback& callback) {
  IVectorPtr label = getInput(*getLabelLayer()).ids;
  preOutput_.grad->one();
  preOutput_.grad->softreluDerivative(*preOutput_.value);
  preOutput_.grad->subByBitCode(numClasses_, *label);

  if (biases_ && biases_->getWGrad()) {
    preOutput_.grad->addByBitCodeBackward(
        numClasses_, *label, *biases_->getWGrad());

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i < inputLayers_.size() - 1; ++i) {
    /* Calculate the W-gradient for the current layer */
    MatrixPtr input = getInputValue(i);
    if (weights_[i]->getWGrad()) {
      preOutput_.grad->mulByBitCodeBackwardWeight(
          numClasses_, *label, *weights_[i]->getWGrad(), *input);

      /* Increasing the number of gradient */
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }

    /* Calculate the input layers error */
    MatrixPtr inputGrad = getInputGrad(i);
    if (inputGrad) {
      preOutput_.grad->mulByBitCodeBackwardError(
          numClasses_, *label, *weights_[i]->getW(), *inputGrad);
    }
  }
}

}  // namespace paddle
