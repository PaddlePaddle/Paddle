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

#include "ExpandConvBaseLayer.h"

#include "paddle/utils/Logging.h"
namespace paddle {

bool ExpandConvBaseLayer::init(const LayerMap &layerMap,
                               const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ConvBaseLayer::init(layerMap, parameterMap);

  /* The class fields channels_ and numFilters_ are the same as in the config
   * i.e., channels_ is the for the input and numFilters_ is for the output
   *
   * But in order for the variables in convTrans having the same semantic
   * meaning as in conv, we need to swap channels_ and numFilters here for
   * convTrans, and in other functions too.
   * */

  /* Initialize the projection */
  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    int numFilters = isDeconv_ ? conf.channels() : numFilters_;
    subM_.push_back(numFilters / conf.groups());
    subN_.push_back(conf.output_x() *
                    (conf.has_output_y() ? conf.output_y() : conf.output_x()));
    int channel = isDeconv_ ? numFilters_ : conf.channels();
    subK_.push_back(
        channel * conf.filter_size() *
        (conf.has_filter_size_y() ? conf.filter_size_y() : conf.filter_size()) /
        conf.groups());
    /* Consistent caffe mode for multiple input */
    caffeMode_ = conf.caffe_mode();
  }

  getOutputSize();

  return true;
}

size_t ExpandConvBaseLayer::getOutputSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = ConvBaseLayer::calOutputSize();
  subN_.clear();
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    subN_.push_back(outputH_[i] * outputW_[i]);
  }
  return layerSize;
}

void ExpandConvBaseLayer::resetExpandInput(size_t height, size_t width) {
  Matrix::resizeOrCreate(expandInput_, height, width, false, useGpu_);
}

void ExpandConvBaseLayer::addSharedBias() {
  size_t mapW = getOutputSize() / numFilters_;
  size_t mapH = getOutputValue()->getElementCnt() / mapW;
  MatrixPtr out =
      Matrix::create(getOutputValue()->getData(), mapH, mapW, false, useGpu_);

  Matrix::resizeOrCreate(transOutValue_, mapW, mapH, false, useGpu_);

  out->transpose(transOutValue_, false);  // false means no memory allocation
  transOutValue_->reshape(transOutValue_->getElementCnt() / numFilters_,
                          numFilters_);

  MatrixPtr bias = Matrix::create(biases_->getW()->getData(),
                                  1,
                                  biases_->getW()->getElementCnt(),
                                  false,
                                  useGpu_);
  transOutValue_->addBias(*bias, 1.0f);

  transOutValue_->reshape(mapW, mapH);
  transOutValue_->transpose(out, false);  // false means no memory allocation

  out->clear();
  bias->clear();
}

void ExpandConvBaseLayer::addUnsharedBias() {
  MatrixPtr outValue = getOutputValue();
  MatrixPtr bias = Matrix::create(biases_->getW()->getData(),
                                  1,
                                  biases_->getW()->getElementCnt(),
                                  false,
                                  useGpu_);
  outValue->addBias(*bias, 1.0f);
}

void ExpandConvBaseLayer::expandOneFrame(MatrixPtr image,
                                         size_t startIdx,
                                         int inIdx) {
  int channel = isDeconv_ ? numFilters_ : channels_[inIdx];

  resetExpandInput(subK_[inIdx] * groups_[inIdx], subN_[inIdx]);

  CHECK_EQ(image->getWidth(),
           static_cast<size_t>(imgSizeH_[inIdx] * imgSizeW_[inIdx] * channel));

  real *imgData = image->getData() + startIdx * image->getWidth();
  MatrixPtr imageTmp =
      Matrix::create(imgData,
                     1,
                     imgSizeH_[inIdx] * imgSizeW_[inIdx] * channel,
                     false,
                     useGpu_);
  expandInput_->convExpand(*imageTmp,
                           imgSizeH_[inIdx],
                           imgSizeW_[inIdx],
                           channel,
                           filterSizeY_[inIdx],
                           filterSize_[inIdx],
                           strideY_[inIdx],
                           stride_[inIdx],
                           paddingY_[inIdx],
                           padding_[inIdx],
                           outputH_[inIdx],
                           outputW_[inIdx]);
  imageTmp->clear();
}

void ExpandConvBaseLayer::expandFwdOnce(MatrixPtr image,
                                        MatrixPtr out,
                                        int inIdx,
                                        int startIdx) {
  int subM = subM_[inIdx];
  int subN = subN_[inIdx];
  int subK = subK_[inIdx];

  expandOneFrame(image, startIdx, inIdx);

  int numFilters = isDeconv_ ? channels_[inIdx] : numFilters_;

  real *outData = out->getData() + startIdx * subN * numFilters;

  real *wgtData = weights_[inIdx]->getW()->getData();
  real *expInData = expandInput_->getData();
  for (int g = 0; g < groups_[inIdx]; ++g) {
    MatrixPtr A =
        Matrix::create(wgtData, subM, subK, false, useGpu_);  // mark transpose
    MatrixPtr B = Matrix::create(expInData, subK, subN, false, useGpu_);
    MatrixPtr C = Matrix::create(outData, subM, subN, false, useGpu_);
    C->mul(*A, *B, 1, 1);

    A->clear();
    B->clear();
    C->clear();
    wgtData += subK * subM;
    expInData += subK * subN;
    outData += subM * subN;
  }
}

void ExpandConvBaseLayer::bpropActs(MatrixPtr out,
                                    MatrixPtr image,
                                    int inpIdx) {
  int channel = isDeconv_ ? numFilters_ : channels_[inpIdx];

  int subM = subM_[inpIdx];
  int subN = subN_[inpIdx];
  int subK = subK_[inpIdx];
  size_t batchSize = image->getHeight();

  /* reset the expand-grad memory */
  resetExpandInput(subK * groups_[inpIdx], subN);

  real *localGradData = out->getData();
  real *tgtGradData = image->getData();
  for (size_t n = 0; n < batchSize; n++) {
    real *wgtData = weights_[inpIdx]->getW()->getData();
    real *expandInData = expandInput_->getData();

    for (int g = 0; g < groups_[inpIdx]; g++) {
      // create temporary matrix
      MatrixPtr C = Matrix::create(expandInData, subK, subN, false, useGpu_);
      MatrixPtr B = Matrix::create(localGradData, subM, subN, false, useGpu_);
      MatrixPtr A = Matrix::create(wgtData, subM, subK, true, useGpu_);
      C->mul(*A, *B);  // mul

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
    MatrixPtr vTmp =
        Matrix::create(tgtGradData,
                       1,
                       imgSizeH_[inpIdx] * imgSizeW_[inpIdx] * channel,
                       false,
                       useGpu_);
    vTmp->convShrink(*oneGradTmp,
                     imgSizeH_[inpIdx],
                     imgSizeW_[inpIdx],
                     channel,
                     filterSizeY_[inpIdx],
                     filterSize_[inpIdx],
                     strideY_[inpIdx],
                     stride_[inpIdx],
                     paddingY_[inpIdx],
                     padding_[inpIdx],
                     outputH_[inpIdx],
                     outputW_[inpIdx],
                     1.0f,
                     1.0f);
    vTmp->clear();
    oneGradTmp->clear();

    // move the data-pointer
    tgtGradData += imgSizeH_[inpIdx] * imgSizeW_[inpIdx] * channel;
  }
}

void ExpandConvBaseLayer::bpropWeights(MatrixPtr image,
                                       MatrixPtr out,
                                       int inpIdx) {
  MatrixPtr weightGrad = weights_[inpIdx]->getWGrad();

  int subM = subM_[inpIdx];
  int subN = subN_[inpIdx];
  int subK = subK_[inpIdx];
  size_t batchSize = image->getHeight();
  resetExpandInput(subK * groups_[inpIdx], subN);

  real *gradData = out->getData();

  for (size_t n = 0; n < batchSize; n++) {  // frame by frame
    // expand
    expandOneFrame(image, n, inpIdx);
    real *wGradData = weightGrad->getData();
    real *expandInData = expandInput_->getData();

    // expand-mul one-group by one
    for (int g = 0; g < groups_[inpIdx]; g++) {
      MatrixPtr A = Matrix::create(expandInData, subK, subN, true, useGpu_);
      MatrixPtr B = Matrix::create(gradData, subM, subN, false, useGpu_);
      MatrixPtr C = Matrix::create(wGradData, subM, subK, false, useGpu_);
      C->mul(*B, *A, 1, 1);

      A->clear();
      B->clear();
      C->clear();
      gradData += subM * subN;
      wGradData += subK * subM;
      expandInData += subK * subN;
    }
  }
}

void ExpandConvBaseLayer::bpropSharedBias(MatrixPtr biases, MatrixPtr v) {
  size_t mapW = getOutputSize() / numFilters_;
  size_t mapH = v->getElementCnt() / mapW;
  MatrixPtr vTmp = Matrix::create(v->getData(), mapH, mapW, false, useGpu_);

  Matrix::resizeOrCreate(transOutValue_, mapW, mapH, false, useGpu_);

  vTmp->transpose(transOutValue_, false);  // false means no memory allocation
  transOutValue_->reshape(transOutValue_->getElementCnt() / numFilters_,
                          numFilters_);
  biases->collectBias(*transOutValue_, 1.0f);
}

void ExpandConvBaseLayer::bpropBiases(MatrixPtr v) {
  MatrixPtr biases = Matrix::create(biases_->getWGrad()->getData(),
                                    1,
                                    biases_->getWGrad()->getElementCnt(),
                                    false,
                                    useGpu_);
  if (sharedBiases_) {
    bpropSharedBias(biases, v);
  } else {
    biases->collectBias(*v, 1.0f);
  }
  biases->clear();
}

}  // namespace paddle
