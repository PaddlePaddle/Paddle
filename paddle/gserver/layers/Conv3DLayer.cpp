/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "Conv3DLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(conv3d, Conv3DLayer);

bool Conv3DLayer::init(const LayerMap &layerMap,
                       const ParameterMap &parameterMap) {
  if (!ConvBaseLayer::init(layerMap, parameterMap)) return false;
  int index = 0;
  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    M_.push_back(numFilters_ / conf.groups());
    K_.push_back(filterPixels_[index] * filterChannels_[index]);

    // create a new weight
    size_t height, width;
    width = filterPixels_[index] * filterChannels_[index];
    height = numFilters_;
    CHECK_EQ(parameters_[index]->getSize(), width * height);
    Weight *w = new Weight(height, width, parameters_[index]);
    weights_.emplace_back(w);
    ++index;
  }
  if (biasParameter_.get()) {
    if (sharedBiases_) {
      CHECK_EQ((size_t)numFilters_, biasParameter_->getSize());
      biases_ =
          std::unique_ptr<Weight>(new Weight(numFilters_, 1, biasParameter_));
    } else {
      biases_ =
          std::unique_ptr<Weight>(new Weight(getSize(), 1, biasParameter_));
    }
  }
  return true;
}

size_t Conv3DLayer::getSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  outputH_.clear();
  outputW_.clear();
  outputD_.clear();
  N_.clear();
  size_t layerSize = 0;
  for (size_t i = 0; i < inputLayers_.size(); ++i) {
    outputW_.push_back(outputSize(
        imgSizeW_[i], filterSize_[i], padding_[i], stride_[i], true));
    outputH_.push_back(outputSize(
        imgSizeH_[i], filterSizeY_[i], paddingY_[i], strideY_[i], true));
    outputD_.push_back(outputSize(
        imgSizeD_[i], filterSizeZ_[i], paddingZ_[i], strideZ_[i], true));

    N_.push_back(outputD_[i] * outputH_[i] * outputW_[i]);
    CHECK(layerSize == 0 || N_[i] * size_t(numFilters_) == layerSize);
    layerSize += N_[i] * numFilters_;
  }
  getOutput().setFrameHeight(outputH_[0]);
  getOutput().setFrameWidth(outputW_[0]);
  getOutput().setFrameDepth(outputD_[0]);
  return layerSize;
}

void Conv3DLayer::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  int outWidth = getSize();
  resetOutput(batchSize, outWidth);

  REGISTER_TIMER_INFO("FwdConv3D", getName().c_str());
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    const MatrixPtr &inMat = getInputValue(i);
    const MatrixPtr &outMat = getOutputValue();
    int M = M_[i];
    int N = N_[i];
    int K = K_[i];
    Matrix::resizeOrCreate(colBuf_, K * groups_[i], N, false, useGpu_);
    MatrixPtr wMat = weights_[i]->getW();
    for (int n = 0; n < batchSize; ++n) {
      colBuf_->vol2Col(inMat->getData() + n * inMat->getStride(),
                       channels_[i],
                       imgSizeD_[i],
                       imgSizeH_[i],
                       imgSizeW_[i],
                       filterSizeZ_[i],
                       filterSizeY_[i],
                       filterSize_[i],
                       strideZ_[i],
                       strideY_[i],
                       stride_[i],
                       paddingZ_[i],
                       paddingY_[i],
                       padding_[i]);

      real *outData = outMat->getData() + n * outMat->getStride();
      MatrixPtr outMatSub =
          Matrix::create(outData, groups_[i] * M, N, false, useGpu_);
      for (int g = 0; g < groups_[i]; g++) {
        MatrixPtr wMatSub = wMat->subMatrix(g * M, M);
        MatrixPtr in = colBuf_->subMatrix(g * K, K);
        MatrixPtr out = outMatSub->subMatrix(g * M, M);
        out->mul(*wMatSub, *in, 1.0, 1.0);
      }
    }
  }
  if (nullptr != this->biasParameter_) {
    this->addBias();
  }
  forwardActivation();
}

void Conv3DLayer::backward(const UpdateCallback &callback) {
  backwardActivation();

  if (biases_ && biases_->getWGrad()) {
    bpropBiases();
    biases_->getParameterPtr()->incUpdate(callback);
  }

  REGISTER_TIMER_INFO("BwdConv3D", getName().c_str());
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    if (weights_[i]->getWGrad()) {
      bpropWeights(i);
    }
    if (getInputGrad(i)) {
      bpropData(i);
    }
    weights_[i]->getParameterPtr()->incUpdate(callback);
  }
}

void Conv3DLayer::bpropWeights(int i) {
  int M = M_[i];
  int N = N_[i];
  int K = K_[i];
  const MatrixPtr &inMat = getInputValue(i);
  Matrix::resizeOrCreate(colBuf_, K * groups_[i], N, false, useGpu_);
  MatrixPtr wGradMat = weights_[i]->getWGrad();
  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  for (int n = 0; n < batchSize; ++n) {
    colBuf_->vol2Col(inMat->getData() + n * inMat->getStride(),
                     channels_[i],
                     imgSizeD_[i],
                     imgSizeH_[i],
                     imgSizeW_[i],
                     filterSizeZ_[i],
                     filterSizeY_[i],
                     filterSize_[i],
                     strideZ_[i],
                     strideY_[i],
                     stride_[i],
                     paddingZ_[i],
                     paddingY_[i],
                     padding_[i]);

    real *outGradData =
        getOutputGrad()->getData() + n * getOutputGrad()->getStride();
    MatrixPtr outGradSub =
        Matrix::create(outGradData, groups_[i] * M, N, false, useGpu_);
    for (int g = 0; g < groups_[i]; ++g) {
      MatrixPtr inMatSub = colBuf_->subMatrix(g * K, K);
      MatrixPtr outG = outGradSub->subMatrix(g * M, M);
      MatrixPtr wGradSub = wGradMat->subMatrix(g * M, M);
      wGradSub->mul(*outG, *(inMatSub->getTranspose()), 1.0, 1.0);
    }
  }
}

void Conv3DLayer::bpropData(int i) {
  int M = M_[i];
  int N = N_[i];
  int K = K_[i];
  Matrix::resizeOrCreate(colBuf_, K * groups_[i], N, false, useGpu_);
  MatrixPtr wMat = weights_[i]->getW();
  int batchSize = inputLayers_[0]->getOutputValue()->getHeight();
  for (int n = 0; n < batchSize; ++n) {
    real *outGradData =
        getOutputGrad()->getData() + n * getOutputGrad()->getStride();
    real *preGradData =
        getInputGrad(i)->getData() + n * getInputGrad(i)->getStride();
    MatrixPtr outGradSub =
        Matrix::create(outGradData, M * groups_[i], N, false, useGpu_);
    for (int g = 0; g < groups_[i]; ++g) {
      MatrixPtr wMatSub = wMat->subMatrix(g * M, M);
      MatrixPtr outG = outGradSub->subMatrix(g * M, M);
      MatrixPtr inGradMatSub = colBuf_->subMatrix(g * K, K);
      inGradMatSub->mul(*(wMatSub->getTranspose()), *outG, 1.0, 0.0);
    }
    colBuf_->col2Vol(preGradData,
                     channels_[i],
                     imgSizeD_[i],
                     imgSizeH_[i],
                     imgSizeW_[i],
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

void Conv3DLayer::bpropBiases() {
  MatrixPtr biases = Matrix::create(biases_->getWGrad()->getData(),
                                    1,
                                    biases_->getWGrad()->getElementCnt(),
                                    false,
                                    useGpu_);
  MatrixPtr outGradMat = getOutputGrad();

  if (this->sharedBiases_) {
    biases->collectSharedBias(*outGradMat, 1.0f);
  } else {
    biases->collectBias(*outGradMat, 1.0f);
  }
}

void Conv3DLayer::addBias() {
  MatrixPtr outMat = getOutputValue();
  MatrixPtr bias = Matrix::create(biases_->getW()->getData(),
                                  1,
                                  biases_->getW()->getElementCnt(),
                                  false,
                                  useGpu_);
  if (this->sharedBiases_) {
    outMat->addSharedBias(*(bias), 1.0f);
  } else {
    outMat->addBias(*(bias), 1.0f);
  }
}

}  // namespace paddle
