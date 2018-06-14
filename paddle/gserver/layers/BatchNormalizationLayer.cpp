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

#include "paddle/utils/Stat.h"
#ifdef PADDLE_WITH_CUDA
#include "hl_batch_transpose.h"
#endif
#include "BatchNormalizationLayer.h"

namespace paddle {

REGISTER_LAYER(batch_norm, BatchNormalizationLayer);

bool BatchNormalizationLayer::init(const LayerMap& layerMap,
                                   const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  if (!BatchNormBaseLayer::init(layerMap, parameterMap)) return false;

  return true;
}

void BatchNormalizationLayer::calMeanAndStd(const MatrixPtr& mat) {
  int numSamples = mat->getHeight();
  Matrix::resizeOrCreate(tmpMat_, numSamples, channels_, false, useGpu_);
  savedMean_->zeroMem();
  savedMean_->accumulateColSum(*mat);
  savedMean_->mulScalar(1.0 / numSamples);  // E[x]

  tmpMat_->assign(*mat);
  tmpMat_->square2();
  savedInvVar_->zeroMem();
  savedInvVar_->accumulateColSum(*tmpMat_);
  savedInvVar_->mulScalar(1.0 / numSamples);   // E[x^2]
  savedInvVar_->addSquare(*savedMean_, -1.0);  // E[x^2] - E^2[x]

  // Variance may be small negative value
  // because of the subtraction operation.
  // Here using clipping.
  savedInvVar_->downClip(real(0.0));

  calMovingMeanAndVar();

  savedInvVar_->subScalar(-epsilon_);
  savedInvVar_->sqrt2(*savedInvVar_);
}

void BatchNormalizationLayer::calMovingMeanAndVar() {
  // calculating and saving moving mean and variance
  auto& movingMean = movingMean_->getW();
  auto& movingVar = movingVar_->getW();
  // movingMean =  movingMean * movingAvgFraction_
  //            + savedMean_ * (1 - movingAvgFraction_)
  movingMean->add(*savedMean_, movingAvgFraction_, 1.0 - movingAvgFraction_);
  // movingVar =  movingVar * movingAvgFraction_
  //           + savedInvVar_ * (1 - movingAvgFraction_)
  movingVar->add(*savedInvVar_, movingAvgFraction_, 1.0 - movingAvgFraction_);
}

void BatchNormalizationLayer::setMeanAndStd() {
  savedMean_->copyFrom(*(movingMean_->getW()));
  savedInvVar_->copyFrom(*(movingVar_->getW()));
  savedInvVar_->downClip(real(0.0));

  savedInvVar_->subScalar(-epsilon_);
  savedInvVar_->sqrt2(*savedInvVar_);
}

void BatchNormalizationLayer::expandMat(const MatrixPtr& in, MatrixPtr& out) {
  CHECK_EQ(in->getWidth(), static_cast<size_t>(channels_ * imgPixels_));
  CHECK_EQ(out->getWidth(), static_cast<size_t>(channels_));
  CHECK(!in->isTransposed());
  CHECK(!out->isTransposed());
  if (imgPixels_ == 1) {
    out->assign(*in);
    return;
  }
  size_t batchSize = in->getHeight();
  CHECK_EQ(out->getHeight(), batchSize * imgPixels_);
  if (useGpu_) {
#ifndef PADDLE_WITH_CUDA
    LOG(FATAL) << "paddle is compiled only for cpu";
#else
    batchTranspose(
        in->getData(), out->getData(), imgPixels_, channels_, batchSize);
#endif
  } else {
    for (size_t i = 0; i < batchSize; i++) {
      const MatrixPtr inTmp =
          Matrix::create(in->getData() + i * imgPixels_ * channels_,
                         channels_,
                         imgPixels_,
                         false,
                         useGpu_);
      MatrixPtr outTmp =
          Matrix::create(out->getData() + i * imgPixels_ * channels_,
                         imgPixels_,
                         channels_,
                         false,
                         useGpu_);
      inTmp->transpose(outTmp, false);
    }
  }
}

void BatchNormalizationLayer::shrinkMat(const MatrixPtr& in, MatrixPtr& out) {
  CHECK_EQ(in->getWidth(), static_cast<size_t>(channels_));
  CHECK_EQ(out->getWidth(), static_cast<size_t>(channels_ * imgPixels_));
  size_t batchSize = out->getHeight();
  CHECK(!in->isTransposed());
  CHECK(!out->isTransposed());
  if (imgPixels_ == 1) {
    out->assign(*in);
    return;
  }
  CHECK_EQ(in->getHeight(), static_cast<size_t>(batchSize * imgPixels_));
  if (useGpu_) {
#ifndef PADDLE_WITH_CUDA
    LOG(FATAL) << "paddle is compiled only for cpu";
#else
    batchTranspose(
        in->getData(), out->getData(), channels_, imgPixels_, batchSize);
#endif
  } else {
    for (size_t i = 0; i < batchSize; i++) {
      const MatrixPtr inTmp =
          Matrix::create(in->getData() + i * channels_ * imgPixels_,
                         imgPixels_,
                         channels_,
                         false,
                         useGpu_);
      MatrixPtr outTmp =
          Matrix::create(out->getData() + i * imgPixels_ * channels_,
                         channels_,
                         imgPixels_,
                         useGpu_);
      inTmp->transpose(outTmp, false);
    }
  }
}

void BatchNormalizationLayer::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = getInputValue(0)->getHeight();
  calFeatureMapSize();
  resetOutput(batchSize, getInputValue(0)->getWidth());

  // for testing in training peroid.
  useGlobalStats_ = (passType == PASS_TEST);
  if (passType == PASS_TEST && config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }

  Matrix::resizeOrCreate(
      expandedIn_, batchSize * imgPixels_, channels_, false, useGpu_);
  Matrix::resizeOrCreate(
      normIn_, batchSize * imgPixels_, channels_, false, useGpu_);
  Matrix::resizeOrCreate(
      expandedOut_, batchSize * imgPixels_, channels_, false, useGpu_);
  expandMat(getInputValue(0), expandedIn_);

  if (useGlobalStats_) {
    if (firstTest_) {
      setMeanAndStd();
      firstTest_ = false;
    }
  } else {
    calMeanAndStd(expandedIn_);
    firstTest_ = true;
  }

  normIn_->assign(*expandedIn_);
  normIn_->addBias(*savedMean_, -1);     // subtract mean.
  normIn_->divRowVector(*savedInvVar_);  // divide std.

  expandedOut_->assign(*normIn_);
  expandedOut_->mulRowVector(*weight_->getW());  // multiple gamma.
  if (biases_) {
    expandedOut_->addBias(*(biases_->getW()), 1);  // add beta.
  }
  MatrixPtr out = getOutputValue();
  shrinkMat(expandedOut_, out);

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void BatchNormalizationLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }
  int batchSize = getInputValue(0)->getHeight();

  Matrix::resizeOrCreate(meanGrad_, 1, channels_, false, useGpu_);
  Matrix::resizeOrCreate(stdGrad_, 1, channels_, false, useGpu_);

  Matrix::resizeOrCreate(
      expandedInGrad_, batchSize * imgPixels_, channels_, false, useGpu_);
  Matrix::resizeOrCreate(
      inGrad_, batchSize, imgPixels_ * channels_, false, useGpu_);
  Matrix::resizeOrCreate(
      normInGrad_, batchSize * imgPixels_, channels_, false, useGpu_);
  Matrix::resizeOrCreate(
      expandedOutGrad_, batchSize * imgPixels_, channels_, false, useGpu_);
  Matrix::resizeOrCreate(
      tmpMat_, batchSize * imgPixels_, channels_, false, useGpu_);
  Matrix::resizeOrCreate(
      tmpGrad_, batchSize * imgPixels_, channels_, false, useGpu_);

  expandMat(getOutputGrad(), expandedOutGrad_);

  // compute derivatives.
  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("BpBiasTimer", getName().c_str());
    biases_->getWGrad()->collectBias(*expandedOutGrad_, 1);
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }
  if (weight_->getWGrad()) {
    tmpMat_->dotMul(*expandedOutGrad_, *normIn_);
    weight_->getWGrad()->collectBias(*tmpMat_, 1);
  }

  // compute input gradients.
  normInGrad_->assign(*expandedOutGrad_);
  normInGrad_->mulRowVector(*(weight_->getW()));  // multiple gamma.
  // normInGrad * (x - \mu)/ \sqrt(\delta^2)
  tmpMat_->dotMul(*normInGrad_, *normIn_);
  stdGrad_->zeroMem();
  stdGrad_->collectBias(*tmpMat_, -1.0 / (batchSize * imgPixels_));
  tmpGrad_->assign(*normIn_);
  tmpGrad_->mulRowVector(*stdGrad_);

  meanGrad_->zeroMem();
  meanGrad_->collectBias(*normInGrad_, -1.0 / (batchSize * imgPixels_));

  expandedInGrad_->zeroMem();
  expandedInGrad_->add(*normInGrad_, *tmpGrad_);
  expandedInGrad_->addRowVector(*meanGrad_);
  expandedInGrad_->divRowVector(*savedInvVar_);

  shrinkMat(expandedInGrad_, inGrad_);
  if (getInputGrad(0)) {
    getInputGrad(0)->add(*getInputGrad(0), *inGrad_);
  }
  {
    REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
    weight_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
