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

#include "Layer.h"
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"

namespace paddle {
/**
 * @brief A layer for normalize the conv layer's output.
 * - Input: One and only one input layer are accepted. The input layer must be
 *        be a data output layer.
 * - Output: The normalized data of the input data.
 * Reference:
 *    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
 *    Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector
 */

class NormalizeLayer : public Layer {
public:
  explicit NormalizeLayer(const LayerConfig& config) : Layer(config) {}
  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);
  void backward(const UpdateCallback& callback);

protected:
  size_t channels_;
  std::unique_ptr<Weight> scale_;
  MatrixPtr scaleDiff_;
  MatrixPtr normBuffer_;
  MatrixPtr dataBuffer_;
  MatrixPtr channelBuffer_;
  MatrixPtr spatialBuffer_;
  MatrixPtr sampleBuffer_;
};

bool NormalizeLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  CHECK(parameters_[0]);
  channels_ = config_.num_filters();
  scale_.reset(new Weight(channels_, 1, parameters_[0]));
  return true;
}

void NormalizeLayer::forward(PassType passType) {
  Layer::forward(passType);
  auto in = getInput(0);
  MatrixPtr inV = getInputValue(0);

  size_t batchSize = inV->getHeight();
  size_t dataDim = inV->getWidth();
  CHECK_EQ(getSize(), dataDim);

  reserveOutput(batchSize, dataDim);
  MatrixPtr outV = getOutputValue();
  size_t spatialDim = dataDim / channels_;

  Matrix::resizeOrCreate(dataBuffer_, batchSize, dataDim, false, useGpu_);
  Matrix::resizeOrCreate(spatialBuffer_, 1, spatialDim, false, useGpu_);
  Matrix::resizeOrCreate(channelBuffer_, channels_, 1, false, useGpu_);
  Matrix::resizeOrCreate(sampleBuffer_, channels_, spatialDim, false, useGpu_);
  Matrix::resizeOrCreate(normBuffer_, batchSize, spatialDim, false, useGpu_);
  normBuffer_->zeroMem();
  spatialBuffer_->zeroMem();
  sampleBuffer_->zeroMem();
  dataBuffer_->zeroMem();
  // add eps to avoid overflow
  normBuffer_->addScalar(*normBuffer_, 1e-6);
  channelBuffer_->resetOne();
  inV->square2(*dataBuffer_);
  for (size_t i = 0; i < batchSize; i++) {
    spatialBuffer_->zeroMem();
    MatrixPtr inTmp = Matrix::create(
        inV->getData() + i * dataDim, channels_, spatialDim, false, useGpu_);
    MatrixPtr dataTmp = Matrix::create(dataBuffer_->getData() + i * dataDim,
                                       channels_,
                                       spatialDim,
                                       false,
                                       useGpu_);
    MatrixPtr outTmp = Matrix::create(
        outV->getData() + i * dataDim, channels_, spatialDim, false, useGpu_);
    MatrixPtr normTmp = Matrix::create(
        normBuffer_->getData() + i * spatialDim, 1, spatialDim, false, useGpu_);
    // compute norm.
    spatialBuffer_->sumCols(*dataTmp, 1, 1);
    spatialBuffer_->sqrt2(*spatialBuffer_);
    normTmp->copyFrom(*spatialBuffer_);
    sampleBuffer_->mul(*channelBuffer_, *spatialBuffer_, 1., 0.);
    sampleBuffer_->dotDiv(*inTmp, *sampleBuffer_);
    outTmp->copyFrom(*sampleBuffer_);

    // scale the layer.
    spatialBuffer_->resetOne();
    sampleBuffer_->mul(*scale_->getW(), *spatialBuffer_, 1., 0.);
    outTmp->dotMul(*outTmp, *sampleBuffer_);
  }
}

void NormalizeLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inG = getInputGrad(0);
  MatrixPtr inV = getInputValue(0);
  MatrixPtr outG = getOutputGrad();
  MatrixPtr outV = getOutputValue();

  auto in = getInput(0);
  size_t batchSize = inG->getHeight();
  size_t dataDim = inG->getWidth();
  size_t spatialDim = dataDim / channels_;

  bool syncFlag = hl_get_sync_flag();
  dataBuffer_->dotMul(*outG, *outV);
  Matrix::resizeOrCreate(scaleDiff_, channels_, 1, false, useGpu_);
  scaleDiff_->zeroMem();
  for (size_t i = 0; i < batchSize; i++) {
    spatialBuffer_->zeroMem();
    channelBuffer_->zeroMem();
    // propagate to param.
    MatrixPtr dataBufferTmp =
        Matrix::create(dataBuffer_->getData() + i * dataDim,
                       channels_,
                       spatialDim,
                       false,
                       useGpu_);
    const MatrixPtr inValueTmp = Matrix::create(
        inV->getData() + i * dataDim, channels_, spatialDim, false, useGpu_);
    const MatrixPtr outGradTmp = Matrix::create(
        outG->getData() + i * dataDim, channels_, spatialDim, false, useGpu_);
    MatrixPtr inGradTmp = Matrix::create(
        inG->getData() + i * dataDim, channels_, spatialDim, false, useGpu_);
    const MatrixPtr normTmp = Matrix::create(
        normBuffer_->getData() + i * spatialDim, 1, spatialDim, false, useGpu_);
    channelBuffer_->sumRows(*dataBufferTmp, 1, 1);
    channelBuffer_->dotDiv(*channelBuffer_, *(scale_->getW()));
    // store a / scale[i] in scaleDiff_ temporary
    scaleDiff_->add(*channelBuffer_, 1.);

    sampleBuffer_->dotMul(*inValueTmp, *outGradTmp);
    spatialBuffer_->sumCols(*sampleBuffer_, 1., 1.);
    // scale the grad
    channelBuffer_->resetOne();
    sampleBuffer_->mul(*channelBuffer_, *spatialBuffer_, 1., 0.);

    inGradTmp->dotMul(*inValueTmp, *sampleBuffer_);
    // divide by square of norm
    spatialBuffer_->dotMul(*normTmp, *normTmp);
    sampleBuffer_->mul(*channelBuffer_, *spatialBuffer_, 1., 0.);
    inGradTmp->dotDiv(*inGradTmp, *sampleBuffer_);
    // subtract
    inGradTmp->add(*outGradTmp, -1, 1);
    // divide by norm
    sampleBuffer_->mul(*channelBuffer_, *normTmp, 1., 0.);
    inGradTmp->dotDiv(*inGradTmp, *sampleBuffer_);
    // scale the diff
    spatialBuffer_->resetOne();
    sampleBuffer_->mul(*scale_->getW(), *spatialBuffer_, 1., 0.);
    inGradTmp->dotMul(*inGradTmp, *sampleBuffer_);
  }
  // updata scale
  if (scale_->getWGrad()) scale_->getWGrad()->copyFrom(*scaleDiff_);
  hl_set_sync_flag(false);
  hl_set_sync_flag(syncFlag);
  scale_->getParameterPtr()->incUpdate(callback);
}
REGISTER_LAYER(normalize, NormalizeLayer);

}  // namespace paddle
