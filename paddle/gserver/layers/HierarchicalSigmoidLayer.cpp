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
                         false);
  Matrix::resizeOrCreate(preOutput_.grad,
                         batchSize,
                         codeLength_,
                         /* trans */ false,
                         false);
  IVectorPtr label = getInput(*getLabelLayer()).ids;
  preOutput_.value->zeroMem();

  if (useGpu_) {
    Matrix::resizeOrCreate(cpuOutput_,
                           output_.value->getHeight(),
                           output_.value->getWidth(),
                           /* trans */ false,
                           false);
    IVector::resizeOrCreate(cpuLabel_, label->getSize(), false);
    cpuLabel_->copyFrom(*label);
    cpuOutput_->copyFrom(*output_.value);
  } else {
    cpuOutput_ = output_.value;
    cpuLabel_ = label;
  }
  /* add the bias-vector */
  if (biases_.get() != NULL) {
    if (useGpu_) {
      Matrix::resizeOrCreate(cpuBias_,
                             1,
                             numClasses_ - 1,
                             /* trans */ false,
                             false);
      cpuBias_->copyFrom(*biases_->getW());
    } else {
      cpuBias_ = biases_->getW();
    }
    preOutput_.value->addByBitCode(numClasses_, *cpuLabel_, *cpuBias_);
  }
  for (size_t i = 0; i < inputLayers_.size() - 1; ++i) {
    MatrixPtr input = getInputValue(i);
    if (useGpu_) {
      Matrix::resizeOrCreate(cpuInput_,
                             input->getHeight(),
                             input->getWidth(),
                             /* trans */ false,
                             false);
      Matrix::resizeOrCreate(cpuWeight_,
                             weights_[i]->getW()->getHeight(),
                             weights_[i]->getW()->getWidth(),
                             /* trans */ false,
                             false);
      cpuInput_->copyFrom(*input);
      cpuWeight_->copyFrom(*weights_[i]->getW());
    } else {
      cpuInput_ = input;
      cpuWeight_ = weights_[i]->getW();
    }
    preOutput_.value->mulByBitCode(
        numClasses_, *cpuLabel_, *cpuWeight_, *cpuInput_);
  }
  // keep consistent with the clipping in the following softrelu
  preOutput_.value->clip(-40.0, 40.0);
  preOutput_.value->sumByBitCode(numClasses_,
                                 *cpuLabel_,
                                 *cpuOutput_,
                                 -1);  // scaleSum
  preOutput_.value->softrelu(*preOutput_.value);
  MatrixPtr sum = Matrix::create(batchSize, 1, /* trans= */ false, false);
  preOutput_.value->rowSum(*sum);
  cpuOutput_->add(*sum);
  if (useGpu_) {
    output_.value->copyFrom(*cpuOutput_);
  } else {
    output_.value = cpuOutput_;
  }
}

void HierarchicalSigmoidLayer::backward(const UpdateCallback& callback) {
  IVectorPtr label = getInput(*getLabelLayer()).ids;
  if (useGpu_) {
    IVector::resizeOrCreate(cpuLabel_, label->getSize(), false);
    cpuLabel_->copyFrom(*label);
  } else {
    cpuLabel_ = label;
  }
  preOutput_.grad->one();
  preOutput_.grad->softreluDerivative(*preOutput_.value);
  preOutput_.grad->subByBitCode(numClasses_, *cpuLabel_);

  if (biases_ && biases_->getWGrad()) {
    MatrixPtr biases_grad = biases_->getWGrad();
    if (useGpu_) {
      Matrix::resizeOrCreate(cpuBias_,
                             1,
                             numClasses_ - 1,
                             /* trans */ false,
                             false);
      cpuBias_->copyFrom(*biases_grad);
    } else {
      cpuBias_ = biases_grad;
    }
    preOutput_.grad->addByBitCodeBackward(numClasses_, *cpuLabel_, *cpuBias_);
    if (useGpu_) {
      biases_grad->copyFrom(*cpuBias_);
    } else {
      biases_grad = cpuBias_;
    }
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i < inputLayers_.size() - 1; ++i) {
    /* Calculate the W-gradient for the current layer */
    MatrixPtr input = getInputValue(i);
    if (weights_[i]->getWGrad()) {
      MatrixPtr weights_grad = weights_[i]->getWGrad();
      if (useGpu_) {
        Matrix::resizeOrCreate(cpuInput_,
                               input->getHeight(),
                               input->getWidth(),
                               /* trans */ false,
                               false);
        Matrix::resizeOrCreate(cpuWeightGrad_,
                               weights_grad->getHeight(),
                               weights_grad->getWidth(),
                               /* trans */ false,
                               false);
        cpuInput_->copyFrom(*input);
        cpuWeightGrad_->copyFrom(*weights_grad);
      } else {
        cpuInput_ = input;
        cpuWeightGrad_ = weights_grad;
      }
      preOutput_.grad->mulByBitCodeBackwardWeight(
          numClasses_, *cpuLabel_, *cpuWeightGrad_, *cpuInput_);
      if (useGpu_) {
        weights_grad->copyFrom(*cpuWeightGrad_);
      } else {
        weights_grad = cpuWeightGrad_;
      }
      /* Increasing the number of gradient */
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }

    /* Calculate the input layers error */
    MatrixPtr inputGrad = getInputGrad(i);
    if (inputGrad) {
      if (useGpu_) {
        Matrix::resizeOrCreate(cpuInputGrad_,
                               inputGrad->getHeight(),
                               inputGrad->getWidth(),
                               /* trans */ false,
                               false);
        Matrix::resizeOrCreate(cpuWeight_,
                               weights_[i]->getW()->getHeight(),
                               weights_[i]->getW()->getWidth(),
                               /* trans */ false,
                               false);
        cpuInputGrad_->copyFrom(*inputGrad);
        cpuWeight_->copyFrom(*weights_[i]->getW());
      } else {
        cpuInputGrad_ = inputGrad;
        cpuWeight_ = weights_[i]->getW();
      }
      preOutput_.grad->mulByBitCodeBackwardError(
          numClasses_, *cpuLabel_, *cpuWeight_, *cpuInputGrad_);
      if (useGpu_) {
        inputGrad->copyFrom(*cpuInputGrad_);
      } else {
        inputGrad = cpuInputGrad_;
      }
    }
  }
}

}  // namespace paddle
