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

#include "DeConv3DLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(deconv3d, DeConv3DLayer);

bool DeConv3DLayer::init(const LayerMap &layerMap,
                         const ParameterMap &parameterMap) {
  if (!ConvBaseLayer::init(layerMap, parameterMap)) return false;
  // for Deconv, the dimension of Kernel is
  // channel * output * depth * height * weigth
  // Matrix storage format: (output * depth * height * weigth) x  channel
  for (int index = 0; index < config_.inputs().size(); ++index) {
    M_.push_back(filterChannels_[index]);
    K_.push_back(filterPixels_[index] * (numFilters_ / groups_[index]));

    // create a new weight
    size_t height, width;
    height = filterPixels_[index] * numFilters_;
    width = filterChannels_[index];
    CHECK_EQ(parameters_[index]->getSize(), width * height);
    Weight *w = new Weight(height, width, parameters_[index]);
    weights_.emplace_back(w);
  }
  if (biasParameter_.get()) {
    if (sharedBiases_) {
      CHECK_EQ((size_t)numFilters_, biasParameter_->getSize());
      biases_ =
          std::unique_ptr<Weight>(new Weight(1, numFilters_, biasParameter_));
    } else {
      biases_ =
          std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
    }
  }
  return true;
}

size_t DeConv3DLayer::getSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  outputH_.clear();
  outputW_.clear();
  outputD_.clear();
  N_.clear();
  NOut_.clear();
  size_t layerSize = 0;
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    outputW_.push_back(
        imageSize(imgSizeW_[i], filterSize_[i], padding_[i], stride_[i], true));
    outputH_.push_back(imageSize(
        imgSizeH_[i], filterSizeY_[i], paddingY_[i], strideY_[i], true));
    outputD_.push_back(imageSize(
        imgSizeD_[i], filterSizeZ_[i], paddingZ_[i], strideZ_[i], true));
    NOut_.push_back(outputD_[i] * outputH_[i] * outputW_[i]);
    N_.push_back(imgSizeD_[i] * imgSizeH_[i] * imgSizeW_[i]);
    CHECK(layerSize == 0 || N_[i] * size_t(numFilters_) == layerSize);
    layerSize += NOut_[i] * numFilters_;
  }
  getOutput().setFrameHeight(outputH_[0]);
  getOutput().setFrameWidth(outputW_[0]);
  getOutput().setFrameDepth(outputD_[0]);
  return layerSize;
}

void DeConv3DLayer::forward(PassType passType) {
  Layer::forward(passType);
  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  int outWidth = getSize();
  resetOutput(batchSize, outWidth);
  const MatrixPtr outMat = getOutputValue();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    REGISTER_TIMER_INFO("FwdDeConv3D", getName().c_str());
    const MatrixPtr &inMat = getInputValue(i);
    int M = M_[i];
    int N = N_[i];
    int K = K_[i];
    MatrixPtr wMat = weights_[i]->getW();
    Matrix::resizeOrCreate(colBuf_, K * groups_[i], N, false, useGpu_);
    for (int n = 0; n < batchSize; ++n) {
      real *inData = inMat->getData() + n * inMat->getStride();
      for (int g = 0; g < groups_[i]; ++g) {
        MatrixPtr inMatSub = Matrix::create(inData, M, N, false, useGpu_);
        MatrixPtr wMatSub = wMat->subMatrix(g * K, K);
        MatrixPtr colBufDataSub = colBuf_->subMatrix(g * K, K);
        colBufDataSub->mul(*wMatSub, *inMatSub, 1.0, 0.0);
        inData += M * N;
      }
      colBuf_->col2Vol(outMat->getData() + n * outMat->getStride(),
                       numFilters_,
                       outputD_[i],
                       outputH_[i],
                       outputW_[i],
                       filterSizeZ_[i],
                       filterSizeY_[i],
                       filterSize_[i],
                       strideZ_[i],
                       strideY_[i],
                       stride_[i],
                       paddingZ_[i],
                       paddingY_[i],
                       padding_[i],
                       1.0,
                       1.0);
    }
  }
  if (nullptr != this->biasParameter_) {
    REGISTER_TIMER_INFO("FwBiasTimer", getName().c_str());
    this->addBias();
  }
  forwardActivation();
}

void DeConv3DLayer::backward(const UpdateCallback &callback) {
  backwardActivation();
  int batchSize = getOutputGrad()->getHeight();
  if (biases_ && biases_->getWGrad()) {
    bpropBiases();
    biases_->getParameterPtr()->incUpdate(callback);
  }
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    if (weights_[i]->getWGrad() || this->needGradient_) {
      int M = M_[i];
      int N = N_[i];
      int K = K_[i];
      REGISTER_TIMER_INFO("BwdDeConv3D", getName().c_str());
      Matrix::resizeOrCreate(colBuf_, K * groups_[i], N, false, useGpu_);
      const MatrixPtr &inMat = getInputValue(i);
      for (int n = 0; n < batchSize; ++n) {
        colBuf_->vol2Col(
            getOutputGrad()->getData() + n * getOutputGrad()->getStride(),
            numFilters_,
            outputD_[i],
            outputH_[i],
            outputW_[i],
            filterSizeZ_[i],
            filterSizeY_[i],
            filterSize_[i],
            strideZ_[i],
            strideY_[i],
            stride_[i],
            paddingZ_[i],
            paddingY_[i],
            padding_[i]);
        if (weights_[i]->getWGrad()) {
          real *inData = inMat->getData() + n * inMat->getStride();
          for (int g = 0; g < groups_[i]; ++g) {
            MatrixPtr colBufDataSub = colBuf_->subMatrix(g * K, K);
            MatrixPtr wGradMatSub =
                weights_[i]->getWGrad()->subMatrix(g * K, K);
            MatrixPtr inMatSub = Matrix::create(inData, M, N, false, useGpu_);
            wGradMatSub->mul(
                *colBufDataSub, *(inMatSub->getTranspose()), 1.0, 1.0);
            inData += M * N;
          }
        }
        if (getInputGrad(i)) {
          real *preGrad =
              getInputGrad(i)->getData() + n * getInputGrad(i)->getStride();
          for (int g = 0; g < groups_[i]; ++g) {
            MatrixPtr w = weights_[i]->getW()->subMatrix(g * K, K);
            MatrixPtr outGradMat = colBuf_->subMatrix(g * K, K);
            MatrixPtr inGradMatSub =
                Matrix::create(preGrad, M, N, false, useGpu_);
            inGradMatSub->mul(*(w->getTranspose()), *outGradMat, 1.0, 1.0);
            preGrad += M * N;
          }
        }
      }
      REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
      weights_[i]->getParameterPtr()->incUpdate(callback);
    }
  }
}
void DeConv3DLayer::bpropWeights(int i) {}
void DeConv3DLayer::bpropData(int i) {}

void DeConv3DLayer::bpropBiases() {
  const MatrixPtr &outGradMat = getOutputGrad();

  if (this->sharedBiases_) {
    biases_->getWGrad()->collectSharedBias(*outGradMat, 1.0f);
  } else {
    biases_->getWGrad()->collectBias(*outGradMat, 1.0f);
  }
}

void DeConv3DLayer::addBias() {
  MatrixPtr outMat = getOutputValue();
  if (this->sharedBiases_) {
    outMat->addSharedBias(*(biases_->getW()), 1.0f);
  } else {
    outMat->addBias(*(biases_->getW()), 1.0f);
  }
}

}  // namespace paddle
