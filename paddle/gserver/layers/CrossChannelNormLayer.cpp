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
  dataBuffer_->zeroMem();
  // add eps to avoid overflow
  normBuffer_->addScalar(*normBuffer_, 1e-6);
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
    outTmp->copyFrom(*inTmp);
    outTmp->divRowVector(*spatialBuffer_);
    // scale the layer.
    outTmp->mulColVector(*scale_->getW());
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
    inGradTmp->copyFrom(*inValueTmp);
    inGradTmp->mulRowVector(*spatialBuffer_);
    // divide by square of norm
    spatialBuffer_->dotMul(*normTmp, *normTmp);
    inGradTmp->divRowVector(*spatialBuffer_);
    // subtract
    inGradTmp->add(*outGradTmp, -1, 1);
    // divide by norm
    inGradTmp->divRowVector(*normTmp);
    // scale the diff
    inGradTmp->mulColVector(*scale_->getW());
  }
  // updata scale
  if (scale_->getWGrad()) scale_->getWGrad()->copyFrom(*scaleDiff_);
  scale_->getParameterPtr()->incUpdate(callback);
}

}  // namespace paddle
