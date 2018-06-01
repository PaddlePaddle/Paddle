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
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"

namespace paddle {
/**
 * @brief A layer for generating priorbox locations and variances.
 * - Input: Two and only two input layer are accepted. The input layer must be
 *          be a data output layer and a convolution output layer.
 * - Output: The priorbox locations and variances of the input data.
 * Reference:
 *    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
 *    Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector
 */

class PriorBoxLayer : public Layer {
 public:  // NOLINT
  explicit PriorBoxLayer(const LayerConfig& config) : Layer(config) {}
  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override {}

 protected:  // NOLINT
  int numPriors_;
  std::vector<int> minSize_;
  std::vector<int> maxSize_;
  std::vector<real> aspectRatio_;
  std::vector<real> variance_;
  MatrixPtr buffer_;
};

REGISTER_LAYER(priorbox, PriorBoxLayer);

bool PriorBoxLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  auto pbConf = config_.inputs(0).priorbox_conf();
  std::vector<real> tmp;
  aspectRatio_.push_back(1.);
  std::copy(pbConf.min_size().begin(),
            pbConf.min_size().end(),
            std::back_inserter(minSize_));
  std::copy(pbConf.max_size().begin(),
            pbConf.max_size().end(),
            std::back_inserter(maxSize_));
  std::copy(pbConf.variance().begin(),
            pbConf.variance().end(),
            std::back_inserter(variance_));
  std::copy(pbConf.aspect_ratio().begin(),
            pbConf.aspect_ratio().end(),
            std::back_inserter(tmp));

  if (maxSize_.size() > 0) CHECK_EQ(minSize_.size(), maxSize_.size());

  // flip aspect ratios
  for (unsigned index = 0; index < tmp.size(); index++) {
    real ar = tmp[index];
    if (fabs(ar - 1.) < 1e-6) continue;
    aspectRatio_.push_back(ar);
    aspectRatio_.push_back(1. / ar);
  }

  numPriors_ = aspectRatio_.size() * minSize_.size() + maxSize_.size();

  return true;
}

void PriorBoxLayer::forward(PassType passType) {
  Layer::forward(passType);
  auto input = getInput(0);
  int layerWidth = input.getFrameWidth();
  int layerHeight = input.getFrameHeight();

  auto image = getInput(1);
  int imageWidth = image.getFrameWidth();
  int imageHeight = image.getFrameHeight();

  real stepW = static_cast<real>(imageWidth) / layerWidth;
  real stepH = static_cast<real>(imageHeight) / layerHeight;
  int dim = layerHeight * layerWidth * numPriors_ * 4;
  reserveOutput(1, dim * 2);
  // use a cpu buffer to compute
  Matrix::resizeOrCreate(buffer_, 1, dim * 2, false, false);
  auto* tmpPtr = buffer_->getData();

  int idx = 0;
  for (int h = 0; h < layerHeight; ++h) {
    for (int w = 0; w < layerWidth; ++w) {
      real centerX = (w + 0.5) * stepW;
      real centerY = (h + 0.5) * stepH;
      for (size_t s = 0; s < minSize_.size(); s++) {
        real minSize = minSize_[s];
        real boxWidth = minSize;
        real boxHeight = minSize;

        // first prior: aspect_ratio == 1.0, compatible to old logic
        tmpPtr[idx++] = (centerX - boxWidth / 2.) / imageWidth;
        tmpPtr[idx++] = (centerY - boxHeight / 2.) / imageHeight;
        tmpPtr[idx++] = (centerX + boxWidth / 2.) / imageWidth;
        tmpPtr[idx++] = (centerY + boxHeight / 2.) / imageHeight;
        // set the variance.
        for (int t = 0; t < 4; t++) tmpPtr[idx++] = variance_[t];

        if (maxSize_.size() > 0) {
          // square prior with size sqrt(minSize * maxSize)
          real maxSize = maxSize_[s];
          boxWidth = boxHeight = sqrt(minSize * maxSize);
          tmpPtr[idx++] = (centerX - boxWidth / 2.) / imageWidth;
          tmpPtr[idx++] = (centerY - boxHeight / 2.) / imageHeight;
          tmpPtr[idx++] = (centerX + boxWidth / 2.) / imageWidth;
          tmpPtr[idx++] = (centerY + boxHeight / 2.) / imageHeight;
          // set the variance.
          for (int t = 0; t < 4; t++) tmpPtr[idx++] = variance_[t];
        }

        // priors with different aspect ratios
        for (size_t r = 0; r < aspectRatio_.size(); r++) {
          real ar = aspectRatio_[r];
          if (fabs(ar - 1.0) < 1e-6) {
            continue;
          }
          boxWidth = minSize * sqrt(ar);
          boxHeight = minSize / sqrt(ar);
          tmpPtr[idx++] = (centerX - boxWidth / 2.) / imageWidth;
          tmpPtr[idx++] = (centerY - boxHeight / 2.) / imageHeight;
          tmpPtr[idx++] = (centerX + boxWidth / 2.) / imageWidth;
          tmpPtr[idx++] = (centerY + boxHeight / 2.) / imageHeight;
          // set the variance.
          for (int t = 0; t < 4; t++) tmpPtr[idx++] = variance_[t];
        }
      }
    }
  }

  // clip the prior's coordidate such that it is within [0, 1]
  for (int d = 0; d < dim * 2; ++d)
    if ((d % 8) < 4)
      tmpPtr[d] = std::min(std::max(tmpPtr[d], (real)0.), (real)1.);
  MatrixPtr outV = getOutputValue();
  outV->copyFrom(buffer_->data_, dim * 2);
}

}  // namespace paddle
