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
  normBuffer_->zeroMem();
  // add eps to avoid overflow
  normBuffer_->addScalar(*normBuffer_, 1e-6);
  inV->square2(*dataBuffer_);
  for (size_t i = 0; i < batchSize; i++) {
    const MatrixPtr inVTmp = createSampleMatrix(inV, i, spatialDim);
    const MatrixPtr dataTmp = createSampleMatrix(dataBuffer_, i, spatialDim);
    MatrixPtr outVTmp = createSampleMatrix(outV, i, spatialDim);
    MatrixPtr normTmp = createSpatialMatrix(normBuffer_, i, spatialDim);

    // compute norm.
    spatialBuffer_->sumCols(*dataTmp, 1, 0);
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
    spatialBuffer_->sumCols(*sampleBuffer_, 1., 1.);
    // scale the grad
    inGTmp->copyFrom(*inVTmp);
    inGTmp->mulRowVector(*spatialBuffer_);
    // divide by square of norm
    spatialBuffer_->dotMul(*normTmp, *normTmp);
    inGTmp->divRowVector(*spatialBuffer_);
    // subtract
    inGTmp->add(*outGTmp, -1, 1);
    // divide by norm
    inGTmp->divRowVector(*normTmp);
    // scale the diff
    inGTmp->mulColVector(*scale_->getW());
  }
  // updata scale
  if (scale_->getWGrad()) scale_->getWGrad()->copyFrom(*scaleDiff_);
  scale_->getParameterPtr()->incUpdate(callback);
}

}  // namespace paddle
