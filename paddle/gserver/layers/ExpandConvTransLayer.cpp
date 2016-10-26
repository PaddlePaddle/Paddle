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
#include "ExpandConvTransLayer.h"

/* The implementation of the convTransLayer is basically a swap of forward and
 * backward of the original convLayer.
 * The variable naming follows the convention of the convLayer.
 * */

namespace paddle {

REGISTER_LAYER(exconvt, ExpandConvTransLayer);

bool ExpandConvTransLayer::init(const LayerMap &layerMap,
                           const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ConvTransBaseLayer::init(layerMap, parameterMap);

  /* Initialize the projection */
  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    subM_.push_back(conf.channels() / conf.groups());
    subN_.push_back(conf.output_x() * conf.output_x());
    subK_.push_back(channel_ * conf.filter_size() * conf.filter_size() /
                    conf.groups());
    /* Consistent caffe mode for multiple input */
    caffeMode_ = conf.caffe_mode();
  }

  return true;
}

// Why this is necessary after calling init?
size_t ExpandConvTransLayer::getSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  imgSizeH_.clear();
  imgSizeW_.clear();
  outputH_.clear();
  outputW_.clear();
  subN_.clear();
  size_t layerSize = 0;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    outputH_.push_back(inputLayers_[i]->getOutput().getFrameHeight());
    outputW_.push_back(inputLayers_[i]->getOutput().getFrameWidth());
    if (outputH_[i] == 0) outputH_[i] = outputX_[i];
    if (outputW_[i] == 0) outputW_[i] = outputX_[i];
    imgSizeH_.push_back(
        imageSize(outputH_[i], filterSize_[i], padding_[i], stride_[i]));
    imgSizeW_.push_back(
        imageSize(outputW_[i], filterSize_[i], padding_[i], stride_[i]));
    subN_.push_back(outputH_[i] * outputW_[i]);
    CHECK(layerSize == 0 ||
            imgSizeH_[i] * imgSizeW_[i] * (size_t)channel_ == layerSize);
    layerSize = imgSizeH_[i] * imgSizeW_[i] * channel_;
  }
  getOutput().setFrameHeight(imgSizeH_[0]);
  getOutput().setFrameWidth(imgSizeW_[0]);
  return layerSize;
}

void ExpandConvTransLayer::resetExpandInput(size_t height, size_t width) {
  Matrix::resizeOrCreate(expandInput_, height, width, false, useGpu_);
}

/*void ExpandConvTransLayer::resetConvOutput(size_t batchSize, int inIdx) {
  Matrix::resizeOrCreate(transOutValue_, batchSize * numFilters_, subN_[inIdx],
                         false, useGpu_);
}*/


void ExpandConvTransLayer::addSharedBias() {
  size_t mapW = getSize() / channel_;
  size_t mapH = getOutputValue()->getElementCnt() / mapW;
  MatrixPtr out =
      Matrix::create(getOutputValue()->getData(), mapH, mapW, false, useGpu_);

  Matrix::resizeOrCreate(transOutValue_, mapW, mapH, false, useGpu_);

  out->transpose(transOutValue_, false);  // false means no memory allocation
  transOutValue_->reshape(transOutValue_->getElementCnt() / channel_,
                          channel_);

  MatrixPtr bias =
      Matrix::create(biases_->getW()->getData(), 1,
                     biases_->getW()->getElementCnt(), false, useGpu_);
  transOutValue_->addBias(*bias, 1.0f);

  transOutValue_->reshape(mapW, mapH);
  transOutValue_->transpose(out, false);  // false means no memory allocation

  out->clear();
  bias->clear();
}

void ExpandConvTransLayer::addUnsharedBias() {
  MatrixPtr outValue = getOutputValue();
  MatrixPtr bias =
      Matrix::create(biases_->getW()->getData(), 1,
                     biases_->getW()->getElementCnt(), false, useGpu_);
  outValue->addBias(*bias, 1.0f);
}


void ExpandConvTransLayer::expandOneFrame(MatrixPtr image, size_t startIdx,
                                     int inIdx) {
  resetExpandInput(subK_[inIdx] * groups_[inIdx], subN_[inIdx]);
  real *imgData = image->getData() + startIdx * image->getWidth();
  MatrixPtr imageTmp = Matrix::create(
      imgData, 1, imgSizeH_[inIdx] * imgSizeW_[inIdx] * channel_, false,
      useGpu_);
  expandInput_->convExpand(*imageTmp, imgSizeH_[inIdx], imgSizeW_[inIdx],
                           channel_, filterSize_[inIdx],
                           filterSize_[inIdx], stride_[inIdx], stride_[inIdx],
                           padding_[inIdx], padding_[inIdx],
                           outputH_[inIdx], outputW_[inIdx]);
  imageTmp->clear();
}

void ExpandConvTransLayer::expandBackOnce(MatrixPtr imageGrad, int inIdx,
                                        int startIdx) {
  int subM = subM_[inIdx];
  int subN = subN_[inIdx];
  int subK = subK_[inIdx];

  LayerPtr prevLayer = getPrev(inIdx);
  if (NULL == prevLayer->getOutputGrad()) {
    return;
  }

  expandOneFrame(imageGrad, startIdx, inIdx);

  real *outGradData =
      prevLayer -> getOutputGrad()->getData()
                  + startIdx * subN * numFilters_[inIdx];

  real *wgtData = weights_[inIdx]->getW()->getData();
  real *expInData = expandInput_->getData();
  for (int g = 0; g < groups_[inIdx]; ++g) {
    MatrixPtr A =
        Matrix::create(wgtData, subK, subM, true, useGpu_);  // mark transpose
    MatrixPtr B = Matrix::create(expInData, subK, subN, false, useGpu_);
    MatrixPtr C = Matrix::create(outGradData, subM, subN, false, useGpu_);
    C->mul(A, B, 1, 1);

    A->clear();
    B->clear();
    C->clear();
    wgtData += subK * subM;
    expInData += subK * subN;
    outGradData += subM * subN;
  }
}

void ExpandConvTransLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  /* note: one sample correspond to one colum, and the
   *   transOutValue correspond sample to one row */
  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  resetOutput(batchSize, getSize());

  MatrixPtr output = nullptr;
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    LayerPtr prevLayer = getPrev(i);
    output = prevLayer->getOutputValue();
    REGISTER_TIMER_INFO("shrinkFwd", getName().c_str());
    shrinkFwd(output, i);
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

void ExpandConvTransLayer::shrinkFwd(MatrixPtr output, int inpIdx) {
  int subM = subM_[inpIdx];
  int subN = subN_[inpIdx];
  int subK = subK_[inpIdx];

  size_t batchSize = output->getHeight();
  MatrixPtr image = getOutputValue();

  /* reset the expand-grad memory */
  resetExpandInput(subK * groups_[inpIdx], subN);

  real *localData = output->getData();
  real *imageData = image->getData();
  for (size_t n = 0; n < batchSize; n++) {
    real *wgtData = weights_[inpIdx]->getW()->getData();
    real *expandInData = expandInput_->getData();

    for (int g = 0; g < groups_[inpIdx]; g++) {
      // create temporary matrix
      MatrixPtr C = Matrix::create(expandInData, subK, subN, false, useGpu_);
      MatrixPtr B = Matrix::create(localData, subM, subN, false, useGpu_);
      MatrixPtr A = Matrix::create(wgtData, subK, subM, false, useGpu_);
      C->mul(A, B);  // mul

      // clear the temporary matrix
      A->clear();
      B->clear();
      C->clear();

      expandInData += subK * subN;
      localData += subM * subN;
      wgtData += subK * subM;
    }

    // shrink one frame outGrad
    MatrixPtr oneTmp = Matrix::create(
        expandInput_->getData(), subK * groups_[inpIdx], subN, false, useGpu_);
    MatrixPtr vTmp = Matrix::create(
        imageData, 1,
        imgSizeH_[inpIdx] * imgSizeW_[inpIdx] * channel_, false,
        useGpu_);
    vTmp->convShrink(*oneTmp, imgSizeH_[inpIdx], imgSizeW_[inpIdx],
                     channel_, filterSize_[inpIdx],
                     filterSize_[inpIdx], stride_[inpIdx], stride_[inpIdx],
                     padding_[inpIdx], padding_[inpIdx],
                     outputH_[inpIdx], outputW_[inpIdx], 1.0f, 1.0f);
    vTmp->clear();
    oneTmp->clear();

    // move the data-pointer
    imageData += imgSizeH_[inpIdx] * imgSizeW_[inpIdx] * channel_;
  }
}

void ExpandConvTransLayer::bpropSharedBias(MatrixPtr biases, MatrixPtr v) {
  size_t mapW = getSize() / channel_;
  size_t mapH = v->getElementCnt() / mapW;
  MatrixPtr vTmp = Matrix::create(v->getData(), mapH, mapW, false, useGpu_);

  Matrix::resizeOrCreate(transOutValue_, mapW, mapH, false, useGpu_);

  vTmp->transpose(transOutValue_, false);  // false means no memory allocation
  vTmp->reshape(transOutValue_->getElementCnt() / channel_, channel_);
  biases->collectBias(*vTmp, 1.0f);
}

void ExpandConvTransLayer::bpropBiases(MatrixPtr v) {
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

void ExpandConvTransLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  MatrixPtr imageGrad = getOutputGrad();
  if (biases_ && biases_->getWGrad()) {
    bpropBiases(imageGrad);
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    /* First, calculate the input layers error */
    for (size_t off = 0; off < imageGrad->getHeight(); off++) {
        expandBackOnce(imageGrad, i, off);
    }
    if (weights_[i]->getWGrad()) {
      /* Then, calculate the W-gradient for the current layer */
      bpropWeights(imageGrad, i);
      /* Increasing the number of gradient */
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}

void ExpandConvTransLayer::bpropWeights(MatrixPtr v, int inpIdx) {
  MatrixPtr weightGrad = weights_[inpIdx]->getWGrad();
  MatrixPtr outputV = getPrev(inpIdx)->getOutputValue();

  int subM = subM_[inpIdx];
  int subN = subN_[inpIdx];
  int subK = subK_[inpIdx];
  size_t batchSize = outputV->getHeight();
  resetExpandInput(subK * groups_[inpIdx], subN);

  real *outputData = outputV -> getData();

  for (size_t n = 0; n < batchSize; n++) {  // frame by frame
    // expand
    expandOneFrame(v, n, inpIdx);
    real *wGradData = weightGrad->getData();
    real *expandInData = expandInput_->getData();

    // expand-mul one-group by one
    for (int g = 0; g < groups_[inpIdx]; g++) {
      MatrixPtr A = Matrix::create(expandInData, subK, subN, false, useGpu_);
      MatrixPtr B = Matrix::create(outputData, subM, subN, true, useGpu_);
      MatrixPtr C = Matrix::create(wGradData, subK, subM, false, useGpu_);
      C->mul(A, B, 1, 1);

      A->clear();
      B->clear();
      C->clear();
      outputData += subM * subN;
      wGradData += subK * subM;
      expandInData += subK * subN;
    }
  }
}


}  // namespace paddle
