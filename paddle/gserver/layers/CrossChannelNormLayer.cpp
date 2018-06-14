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

#include "Layer.h"
#include "NormLayer.h"
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"

namespace paddle {

MatrixPtr CrossChannelNormLayer::createSampleMatrix(MatrixPtr data,
                                                    size_t iter,
                                                    size_t spatialDim) {
  return Matrix::create(data->getData() + iter * channels_ * spatialDim,
                        channels_,
                        spatialDim,
                        false,
                        useGpu_);
}

MatrixPtr CrossChannelNormLayer::createSpatialMatrix(MatrixPtr data,
                                                     size_t iter,
                                                     size_t spatialDim) {
  return Matrix::create(
      data->getData() + iter * spatialDim, 1, spatialDim, false, useGpu_);
}

bool CrossChannelNormLayer::init(const LayerMap& layerMap,
                                 const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  CHECK(parameters_[0]);
  const NormConfig& conf = config_.inputs(0).norm_conf();
  channels_ = conf.channels();
  scale_.reset(new Weight(channels_, 1, parameters_[0]));
  return true;
}

void CrossChannelNormLayer::forward(PassType passType) {
  Layer::forward(passType);
  MatrixPtr inV = getInputValue(0);

  size_t batchSize = inV->getHeight();
  size_t dataDim = inV->getWidth();
  CHECK_EQ(getSize(), dataDim);

  reserveOutput(batchSize, dataDim);
  MatrixPtr outV = getOutputValue();
  size_t spatialDim = dataDim / channels_;

  Matrix::resizeOrCreate(dataBuffer_, batchSize, dataDim, false, useGpu_);
  Matrix::resizeOrCreate(spatialBuffer_, 1, spatialDim, false, useGpu_);
  Matrix::resizeOrCreate(normBuffer_, batchSize, spatialDim, false, useGpu_);

  inV->square2(*dataBuffer_);
  for (size_t i = 0; i < batchSize; i++) {
    const MatrixPtr inVTmp = createSampleMatrix(inV, i, spatialDim);
    const MatrixPtr dataTmp = createSampleMatrix(dataBuffer_, i, spatialDim);
    MatrixPtr outVTmp = createSampleMatrix(outV, i, spatialDim);
    MatrixPtr normTmp = createSpatialMatrix(normBuffer_, i, spatialDim);

    // compute norm.
    spatialBuffer_->sumCols(*dataTmp, 1, 0);
    // add eps to avoid overflow
    spatialBuffer_->add(1e-6);
    spatialBuffer_->sqrt2(*spatialBuffer_);
    normTmp->copyFrom(*spatialBuffer_);
    outVTmp->copyFrom(*inVTmp);
    outVTmp->divRowVector(*spatialBuffer_);
    // scale the layer.
    outVTmp->mulColVector(*scale_->getW());
  }
}

void CrossChannelNormLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inG = getInputGrad(0);
  MatrixPtr inV = getInputValue(0);
  MatrixPtr outG = getOutputGrad();
  MatrixPtr outV = getOutputValue();

  size_t batchSize = inG->getHeight();
  size_t dataDim = inG->getWidth();
  size_t spatialDim = dataDim / channels_;

  MatrixPtr inGBuffer;
  Matrix::resizeOrCreate(inGBuffer, channels_, spatialDim, false, useGpu_);

  dataBuffer_->dotMul(*outG, *outV);
  Matrix::resizeOrCreate(scaleDiff_, channels_, 1, false, useGpu_);
  Matrix::resizeOrCreate(channelBuffer_, channels_, 1, false, useGpu_);
  Matrix::resizeOrCreate(sampleBuffer_, channels_, spatialDim, false, useGpu_);
  scaleDiff_->zeroMem();
  for (size_t i = 0; i < batchSize; i++) {
    MatrixPtr outGTmp = createSampleMatrix(outG, i, spatialDim);
    const MatrixPtr dataTmp = createSampleMatrix(dataBuffer_, i, spatialDim);
    const MatrixPtr inVTmp = createSampleMatrix(inV, i, spatialDim);
    const MatrixPtr inGTmp = createSampleMatrix(inG, i, spatialDim);
    const MatrixPtr normTmp = createSpatialMatrix(normBuffer_, i, spatialDim);

    channelBuffer_->sumRows(*dataTmp, 1, 0);
    channelBuffer_->dotDiv(*channelBuffer_, *(scale_->getW()));
    // store a / scale[i] in scaleDiff_ temporary
    scaleDiff_->add(*channelBuffer_, 1.);

    sampleBuffer_->dotMul(*inVTmp, *outGTmp);
    spatialBuffer_->sumCols(*sampleBuffer_, 1., 0.);
    // scale the grad
    inGBuffer->copyFrom(*inVTmp);
    inGBuffer->mulRowVector(*spatialBuffer_);
    // divide by square of norm
    spatialBuffer_->dotMul(*normTmp, *normTmp);
    inGBuffer->divRowVector(*spatialBuffer_);
    // subtract
    inGBuffer->add(*outGTmp, -1, 1);
    // divide by norm
    inGBuffer->divRowVector(*normTmp);
    // scale the diff
    inGBuffer->mulColVector(*scale_->getW());

    inGTmp->add(*inGBuffer);
  }
  // updata scale
  if (scale_->getWGrad()) scale_->getWGrad()->add(*scaleDiff_);
  scale_->getParameterPtr()->incUpdate(callback);
}

}  // namespace paddle
