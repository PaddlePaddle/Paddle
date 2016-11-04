/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "ExpandConvLayer.h"

namespace paddle {

REGISTER_LAYER(exconv, ExpandConvLayer);

bool ExpandConvLayer::init(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ConvBaseLayer::init(layerMap, parameterMap);

  /* Initialize the projection */
  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    subM_.push_back(numFilters_ / conf.groups());
    subN_.push_back(conf.output_x() * conf.output_x());
    subK_.push_back(conf.channels() * conf.filter_size() * conf.filter_size() /
                    conf.groups());
    /* Consistent caffe mode for multiple input */
    caffeMode_ = conf.caffe_mode();
  }

  /* initialize the weightList */
  CHECK(inputLayers_.size() == parameters_.size());
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    size_t height, width;
    height = filterPixels_[i] * filterChannels_[i];
    width = numFilters_;

    // create a new weight
    CHECK_EQ(parameters_[i]->getSize(), width * height);
    Weight* w = new Weight(height, width, parameters_[i]);
    weights_.emplace_back(w);
  }

  return true;
}

size_t ExpandConvLayer::getOutputSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = ConvBaseLayer::calOutputSize();
  subN_.clear();
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    subN_.push_back(outputH_[i] * outputW_[i]);
  }
  return layerSize;
}

void ExpandConvLayer::resetExpandInput(size_t height, size_t width) {
  Matrix::resizeOrCreate(expandInput_, height, width, false, useGpu_);
}

void ExpandConvLayer::resetConvOutput(size_t batchSize, int inIdx) {
  Matrix::resizeOrCreate(transOutValue_, batchSize * numFilters_, subN_[inIdx],
                         false, useGpu_);
}

void ExpandConvLayer::expandOneFrame(MatrixPtr image, size_t startIdx,
                                     int inIdx) {
  resetExpandInput(subK_[inIdx] * groups_[inIdx], subN_[inIdx]);
  real *imgData = image->getData() + startIdx * image->getWidth();
  MatrixPtr imageTmp = Matrix::create(
      imgData, 1, imgSizeH_[inIdx] * imgSizeW_[inIdx] * channels_[inIdx], false,
      useGpu_);
  expandInput_->convExpand(*imageTmp, imgSizeH_[inIdx], imgSizeW_[inIdx],
                           channels_[inIdx], filterSize_[inIdx],
                           filterSize_[inIdx], stride_[inIdx], stride_[inIdx],
                           padding_[inIdx], padding_[inIdx],
                           outputH_[inIdx], outputW_[inIdx]);
  imageTmp->clear();
}

void ExpandConvLayer::expandFwdOnce(MatrixPtr image, int inIdx, int startIdx) {
  int subM = subM_[inIdx];
  int subN = subN_[inIdx];
  int subK = subK_[inIdx];

  expandOneFrame(image, startIdx, inIdx);

  real *outData =
      getOutputValue()->getData() + startIdx * subN * numFilters_;

  real *wgtData = weights_[inIdx]->getW()->getData();
  real *expInData = expandInput_->getData();
  for (int g = 0; g < groups_[inIdx]; ++g) {
    MatrixPtr A =
        Matrix::create(wgtData, subK, subM, true, useGpu_);  // mark transpose
    MatrixPtr B = Matrix::create(expInData, subK, subN, false, useGpu_);
    MatrixPtr C = Matrix::create(outData, subM, subN, false, useGpu_);
    C->mul(A, B, 1, 1);

    A->clear();
    B->clear();
    C->clear();
    wgtData += subK * subM;
    expInData += subK * subN;
    outData += subM * subN;
  }
}

void ExpandConvLayer::addSharedBias() {
  size_t mapW = getOutputValue()->getWidth() / numFilters_;
  size_t mapH = getOutputValue()->getElementCnt() / mapW;
  MatrixPtr out =
      Matrix::create(getOutputValue()->getData(), mapH, mapW, false, useGpu_);

  Matrix::resizeOrCreate(transOutValue_, mapW, mapH, false, useGpu_);

  out->transpose(transOutValue_, false);  // false means no memory allocation
  transOutValue_->reshape(transOutValue_->getElementCnt() / numFilters_,
                          numFilters_);

  MatrixPtr bias =
      Matrix::create(biases_->getW()->getData(), 1,
                     biases_->getW()->getElementCnt(), false, useGpu_);
  transOutValue_->addBias(*bias, 1.0f);

  transOutValue_->reshape(mapW, mapH);
  transOutValue_->transpose(out, false);  // false means no memory allocation

  out->clear();
  bias->clear();
}

void ExpandConvLayer::addUnsharedBias() {
  MatrixPtr outValue = getOutputValue();
  MatrixPtr bias =
      Matrix::create(biases_->getW()->getData(), 1,
                     biases_->getW()->getElementCnt(), false, useGpu_);
  outValue->addBias(*bias, 1.0f);
}

void ExpandConvLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  /* note: one sample correspond to one colum, and the
   *   transOutValue correspond sample to one row */
  int batchSize = inputLayers_[0]->getOutputValue()->getWidth();
  batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  resetOutput(batchSize, getOutputSize());

  MatrixPtr image = nullptr;
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    LayerPtr prevLayer = getPrev(i);
    image = prevLayer->getOutputValue();
    for (size_t off = 0; off < image->getHeight(); off++) {
      REGISTER_TIMER_INFO("expandFwdOnce", getName().c_str());
      expandFwdOnce(image, i, off);
    }
  }
  /* add the bias-vector */
  if (biases_.get() != NULL) {
    if (sharedBiases_) {
      addSharedBias();
    } else {
      addUnsharedBias();
    }
  }

  /* activation */
  forwardActivation();
}

void ExpandConvLayer::bpropSharedBias(MatrixPtr biases, MatrixPtr v) {
  size_t mapW = v->getWidth() / numFilters_;
  size_t mapH = v->getElementCnt() / mapW;
  MatrixPtr vTmp = Matrix::create(v->getData(), mapH, mapW, false, useGpu_);

  Matrix::resizeOrCreate(transOutValue_, mapW, mapH, false, useGpu_);

  vTmp->transpose(transOutValue_, false);  // false means no memory allocation
  vTmp->reshape(transOutValue_->getElementCnt() / numFilters_, numFilters_);
  biases->collectBias(*vTmp, 1.0f);
}

void ExpandConvLayer::bpropBiases(MatrixPtr v) {
  MatrixPtr biases =
      Matrix::create(biases_->getWGrad()->getData(), 1,
                     biases_->getWGrad()->getElementCnt(), false, useGpu_);
  if (sharedBiases_) {
    bpropSharedBias(biases, v);
  } else {
    biases->collectBias(*v, 1.0f);
  }
  biases->clear();
}

void ExpandConvLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  MatrixPtr outGrad = getOutputGrad();
  if (biases_ && biases_->getWGrad()) {
    bpropBiases(outGrad);
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    /* First, calculate the input layers error */
    bpropActs(outGrad, i);
    if (weights_[i]->getWGrad()) {
      /* Then, calculate the W-gradient for the current layer */
      bpropWeights(outGrad, i);
      /* Increasing the number of gradient */
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

void ExpandConvLayer::bpropWeights(MatrixPtr v, int inpIdx) {
  MatrixPtr weightGrad = weights_[inpIdx]->getWGrad();
  MatrixPtr inputV = getPrev(inpIdx)->getOutputValue();

  int subM = subM_[inpIdx];
  int subN = subN_[inpIdx];
  int subK = subK_[inpIdx];
  size_t batchSize = inputV->getHeight();
  resetExpandInput(subK * groups_[inpIdx], subN);
  resetConvOutput(batchSize, inpIdx);

  real *gradData = v->getData();

  for (size_t n = 0; n < batchSize; n++) {  // frame by frame
    // expand
    expandOneFrame(inputV, n, inpIdx);
    real *wGradData = weightGrad->getData();
    real *expandInData = expandInput_->getData();

    // expand-mul one-group by one
    for (int g = 0; g < groups_[inpIdx]; g++) {
      MatrixPtr A = Matrix::create(expandInData, subK, subN, false, useGpu_);
      MatrixPtr B = Matrix::create(gradData, subM, subN, true, useGpu_);
      MatrixPtr C = Matrix::create(wGradData, subK, subM, false, useGpu_);
      C->mul(A, B, 1, 1);

      A->clear();
      B->clear();
      C->clear();
      gradData += subM * subN;
      wGradData += subK * subM;
      expandInData += subK * subN;
    }
  }
}

void ExpandConvLayer::bpropActs(MatrixPtr v, int inpIdx) {
  LayerPtr prevLayer = getPrev(inpIdx);
  if (NULL == prevLayer->getOutputGrad()) {
    return;
  }

  int subM = subM_[inpIdx];
  int subN = subN_[inpIdx];
  int subK = subK_[inpIdx];
  size_t batchSize = v->getHeight();
  MatrixPtr tgtGrad = prevLayer->getOutputGrad();

  /* reset the expand-grad memory */
  resetExpandInput(subK * groups_[inpIdx], subN);
  resetConvOutput(batchSize, inpIdx);

  real *localGradData = v->getData();
  real *tgtGradData = tgtGrad->getData();
  for (size_t n = 0; n < batchSize; n++) {
    real *wgtData = weights_[inpIdx]->getW()->getData();
    real *expandInData = expandInput_->getData();

    for (int g = 0; g < groups_[inpIdx]; g++) {
      // create temporary matrix
      MatrixPtr C = Matrix::create(expandInData, subK, subN, false, useGpu_);
      MatrixPtr B = Matrix::create(localGradData, subM, subN, false, useGpu_);
      MatrixPtr A = Matrix::create(wgtData, subK, subM, false, useGpu_);
      C->mul(A, B);  // mul

      // clear the temporary matrix
      A->clear();
      B->clear();
      C->clear();

      expandInData += subK * subN;
      localGradData += subM * subN;
      wgtData += subK * subM;
    }

    // shrink one frame outGrad
    MatrixPtr oneGradTmp = Matrix::create(
        expandInput_->getData(), subK * groups_[inpIdx], subN, false, useGpu_);
    MatrixPtr vTmp = Matrix::create(
        tgtGradData, 1,
        imgSizeH_[inpIdx] * imgSizeW_[inpIdx] * channels_[inpIdx], false,
        useGpu_);
    vTmp->convShrink(*oneGradTmp, imgSizeH_[inpIdx], imgSizeW_[inpIdx],
                     channels_[inpIdx], filterSize_[inpIdx],
                     filterSize_[inpIdx], stride_[inpIdx], stride_[inpIdx],
                     padding_[inpIdx], padding_[inpIdx],
                     outputH_[inpIdx], outputW_[inpIdx], 1.0f, 1.0f);
    vTmp->clear();
    oneGradTmp->clear();

    // move the data-pointer
    tgtGradData += imgSizeH_[inpIdx] * imgSizeW_[inpIdx] * channels_[inpIdx];
  }
}

}  // namespace paddle
