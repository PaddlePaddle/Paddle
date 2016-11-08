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

#include "SpatialPyramidPoolLayer.h"

namespace paddle {

REGISTER_LAYER(spp, SpatialPyramidPoolLayer);

ProjectionConfig SpatialPyramidPoolLayer::getConfig(size_t imgSizeW,
                                                    size_t imgSizeH,
                                                    size_t channels,
                                                    size_t pyramidLevel,
                                                    std::string& poolType) {
  ProjectionConfig config;
  config.set_type("pool");
  PoolConfig* conf = config.mutable_pool_conf();
  conf->set_channels(channels);
  conf->set_img_size(imgSizeW);
  conf->set_img_size_y(imgSizeH);
  conf->set_pool_type(poolType);

  int numBins = std::pow(2, pyramidLevel);

  int sizeH = std::ceil(imgSizeH / static_cast<double>(numBins));
  int remainderH = sizeH * numBins - imgSizeH;
  int paddingH = (remainderH + 1) / 2;
  int outSizeH = outputSize(imgSizeH, sizeH, paddingH, sizeH);

  int sizeW = std::ceil(imgSizeW / static_cast<double>(numBins));
  int remainderW = sizeW * numBins - imgSizeW;
  int paddingW = (remainderW + 1) / 2;
  int outSizeW = outputSize(imgSizeW, sizeW, paddingW, sizeW);

  conf->set_stride(sizeW);
  conf->set_stride_y(sizeH);
  conf->set_size_x(sizeW);
  conf->set_size_y(sizeH);
  conf->set_padding(paddingW);
  conf->set_padding_y(paddingH);
  conf->set_output_x(outSizeW);
  conf->set_output_y(outSizeH);
  config.set_output_size(outSizeH * outSizeW * channels);
  return config;
}

void SpatialPyramidPoolLayer::splitInput(Argument& input, size_t height,
                                         size_t width, bool useGpu) {
  input.value = getInput(0).value;
  if (passType_ != PASS_TEST && needGradient()) {
    Matrix::resizeOrCreate(input.grad, height, width, /* trans */ false,
                           useGpu);
    input.grad->zeroMem();
  }
}

bool SpatialPyramidPoolLayer::init(const LayerMap& layerMap,
                                   const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(config_.inputs_size(), 1);

  const SppConfig& sppConf = config_.inputs(0).spp_conf();
  pyramidHeight_ = sppConf.pyramid_height();
  poolType_ = sppConf.pool_type();

  channels_ = sppConf.channels();
  imgSizeW_ = sppConf.img_size();
  imgSizeH_ = sppConf.has_img_size_y() ? sppConf.img_size_y() : imgSizeW_;
  poolProjections_.reserve(pyramidHeight_);
  projCol_.reserve(pyramidHeight_);
  projInput_.reserve(pyramidHeight_);
  projOutput_.resize(pyramidHeight_);

  size_t startCol = 0;
  size_t endCol = 0;
  for (size_t i = 0; i < pyramidHeight_; i++) {
    poolProjections_.emplace_back(PoolProjection::create(
      getConfig(imgSizeW_, imgSizeH_, channels_, i, poolType_),
      nullptr, useGpu_));
    endCol += poolProjections_[i]->getOutputSize();
    projCol_.push_back(std::make_pair(startCol, endCol));
    startCol = endCol;
    projInput_.emplace_back(Argument());
  }
  CHECK_EQ(endCol, getSize());
  return true;
}

void SpatialPyramidPoolLayer::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = getInput(0).getBatchSize();
  resetOutput(batchSize, getSize());
  for (size_t i = 0; i < pyramidHeight_; i++) {
    size_t startCol = projCol_[i].first;
    size_t endCol = projCol_[i].second;
    projOutput_[i].value = output_.value->subColMatrix(startCol, endCol);
    projOutput_[i].grad = output_.grad->subColMatrix(startCol, endCol);
    splitInput(projInput_[i], getInput(0).value->getHeight(),
               getInput(0).value->getWidth(), useGpu_);
  }
  for (size_t i = 0; i < pyramidHeight_; i++) {
    poolProjections_[i]->forward(&projInput_[i], &projOutput_[i], passType);
  }
}

void SpatialPyramidPoolLayer::backward(const UpdateCallback& callback) {
  for (size_t i = 0; i < pyramidHeight_; i++) {
    if (poolProjections_[i]) {
      poolProjections_[i]->backward(callback);
      getInput(0).grad->add(*projInput_[i].grad);
    }
  }
}

}  // namespace paddle

