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

#include "BlockExpandLayer.h"

#include "paddle/utils/Logging.h"

namespace paddle {

REGISTER_LAYER(blockexpand, BlockExpandLayer);

bool BlockExpandLayer::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(config_.inputs_size(), 1);
  const BlockExpandConfig& blockConf = config_.inputs(0).block_expand_conf();
  blockH_ = blockConf.block_y();
  blockW_ = blockConf.block_x();
  strideH_ = blockConf.stride_y();
  strideW_ = blockConf.stride_x();
  paddingH_ = blockConf.padding_y();
  paddingW_ = blockConf.padding_x();
  channels_ = blockConf.channels();
  imgSizeH_ = blockConf.img_size_y();
  imgSizeW_ = blockConf.img_size_x();

  return true;
}

size_t BlockExpandLayer::getBlockNum() {
  CHECK_EQ(inputLayers_.size(), 1UL);
  const BlockExpandConfig& blockConf = config_.inputs(0).block_expand_conf();
  imgSizeH_ = inputLayers_[0]->getOutput().getFrameHeight();
  imgSizeW_ = inputLayers_[0]->getOutput().getFrameWidth();
  if (imgSizeH_ == 0) {
    imgSizeH_ = blockConf.img_size_y();
  }
  if (imgSizeW_ == 0) {
    imgSizeW_ = blockConf.img_size_x();
  }
  size_t tmpH = 2 * paddingH_ + imgSizeH_ - blockH_;
  outputH_ = (int)tmpH < 0 ? 1 : 1 + (tmpH + strideH_ - 1) / strideH_;
  size_t tmpW = 2 * paddingW_ + imgSizeW_ - blockW_;
  outputW_ = (int)tmpW < 0 ? 1 : 1 + (tmpW + strideW_ - 1) / strideW_;

  return outputH_ * outputW_;
}

void BlockExpandLayer::forward(PassType passType) {
  Layer::forward(passType);

  size_t batchSize = inputLayers_[0]->getOutputValue()->getHeight();

  size_t blockNum = getBlockNum();
  size_t blockSize = blockH_ * blockW_ * channels_;
  resetOutput(blockNum * batchSize, blockSize);
  Argument& out = getOutput();
  MatrixPtr outV = getOutputValue();

  MatrixPtr input = getPrev(0)->getOutputValue();
  Matrix::resizeOrCreate(outVTrans_, blockSize, blockNum, false, useGpu_);
  ICpuGpuVector::resizeOrCreate(
      out.sequenceStartPositions, batchSize + 1, false);
  IVector::resizeOrCreate(out.cpuSequenceDims, 2 * batchSize, false);
  int* start = out.sequenceStartPositions->getMutableData(false);
  int* dims = out.cpuSequenceDims->getData();
  for (size_t i = 0; i < batchSize; i++) {
    outVTrans_->zeroMem();
    /* expand each block as one row */
    MatrixPtr inputTmp =
        Matrix::create(input->getData() + i * input->getWidth(),
                       1,
                       input->getWidth(),
                       false,
                       useGpu_);
    outVTrans_->convExpand(*inputTmp,
                           imgSizeH_,
                           imgSizeW_,
                           channels_,
                           blockH_,
                           blockW_,
                           strideH_,
                           strideW_,
                           paddingH_,
                           paddingW_,
                           outputH_,
                           outputW_);
    MatrixPtr outVTmp =
        Matrix::create(outV->getData() + i * blockNum * blockSize,
                       blockNum,
                       blockSize,
                       false,
                       useGpu_);
    outVTrans_->transpose(outVTmp, false);
    start[i] = i * blockNum;
    dims[2 * i] = outputH_;
    dims[2 * i + 1] = outputW_;
  }
  start[batchSize] = batchSize * blockNum;
}

void BlockExpandLayer::backward(const UpdateCallback& callback) {
  size_t blockNum = outputH_ * outputW_;
  size_t blockSize = blockH_ * blockW_ * channels_;
  /* Calculate the input layers error */
  MatrixPtr preGrad = inputLayers_[0]->getOutputGrad();
  if (!preGrad) {
    return;
  }
  MatrixPtr grad = getOutputGrad();
  MatrixPtr gradTrans = Matrix::create(blockSize, blockNum, false, useGpu_);
  size_t batchSize = preGrad->getHeight();

  CHECK_EQ(batchSize * blockNum, grad->getHeight());
  CHECK_EQ(blockSize, grad->getWidth());

  for (size_t i = 0; i < batchSize; i++) {
    MatrixPtr gradTmp =
        Matrix::create(grad->getData() + i * blockNum * blockSize,
                       blockNum,
                       blockSize,
                       false,
                       useGpu_);
    gradTmp->transpose(gradTrans, false);
    MatrixPtr preGradTmp =
        Matrix::create(preGrad->getData() + i * preGrad->getWidth(),
                       1,
                       preGrad->getWidth(),
                       false,
                       useGpu_);
    preGradTmp->convShrink(*gradTrans,
                           imgSizeH_,
                           imgSizeW_,
                           channels_,
                           blockH_,
                           blockW_,
                           strideH_,
                           strideW_,
                           paddingH_,
                           paddingW_,
                           outputH_,
                           outputW_,
                           1.0,
                           1.0);
  }
}

}  // namespace paddle
