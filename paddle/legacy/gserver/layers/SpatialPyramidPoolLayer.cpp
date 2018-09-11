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
  int paddingH = (sizeH * numBins - imgSizeH + 1) / 2;
  int outSizeH = outputSize(imgSizeH, sizeH, paddingH, sizeH, true);

  int sizeW = std::ceil(imgSizeW / static_cast<double>(numBins));
  int paddingW = (sizeW * numBins - imgSizeW + 1) / 2;
  int outSizeW = outputSize(imgSizeW, sizeW, paddingW, sizeW, true);

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

size_t SpatialPyramidPoolLayer::getSize() {
  CHECK_EQ(inputLayers_.size(), 1UL);
  size_t layerSize = 0;
  const ImageConfig& conf = config_.inputs(0).spp_conf().image_conf();
  imgSizeH_ = inputLayers_[0]->getOutput().getFrameHeight();
  imgSizeW_ = inputLayers_[0]->getOutput().getFrameWidth();
  if (imgSizeH_ == 0) {
    imgSizeH_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  }
  if (imgSizeW_ == 0) {
    imgSizeW_ = conf.img_size();
  }

  size_t outputH = 1;
  size_t outputW = (std::pow(4, pyramidHeight_) - 1) / (4 - 1);

  layerSize = outputH * outputW * channels_;
  return layerSize;
}

bool SpatialPyramidPoolLayer::init(const LayerMap& layerMap,
                                   const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(config_.inputs_size(), 1);

  const SppConfig& sppConf = config_.inputs(0).spp_conf();
  pyramidHeight_ = sppConf.pyramid_height();
  poolType_ = sppConf.pool_type();

  const ImageConfig& imageConf = sppConf.image_conf();
  channels_ = imageConf.channels();
  imgSizeW_ = imageConf.img_size();
  imgSizeH_ = imageConf.has_img_size_y() ? imageConf.img_size_y() : imgSizeW_;
  poolProjections_.reserve(pyramidHeight_);
  projCol_.reserve(pyramidHeight_);
  projOutput_.resize(pyramidHeight_);

  size_t startCol = 0;
  size_t endCol = 0;
  for (size_t i = 0; i < pyramidHeight_; i++) {
    poolProjections_.emplace_back(PoolProjection::create(
        getConfig(imgSizeW_, imgSizeH_, channels_, i, poolType_),
        nullptr,
        useGpu_));
    endCol += poolProjections_[i]->getOutputSize();
    projCol_.push_back(std::make_pair(startCol, endCol));
    startCol = endCol;
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
    if (output_.grad) {
      projOutput_[i].grad = output_.grad->subColMatrix(startCol, endCol);
    }
  }
  for (size_t i = 0; i < pyramidHeight_; i++) {
    poolProjections_[i]->forward(&getInput(0), &projOutput_[i], passType);
  }
}

void SpatialPyramidPoolLayer::backward(const UpdateCallback& callback) {
  for (size_t i = 0; i < pyramidHeight_; i++) {
    if (poolProjections_[i]) {
      poolProjections_[i]->backward(callback);
    }
  }
}

}  // namespace paddle
