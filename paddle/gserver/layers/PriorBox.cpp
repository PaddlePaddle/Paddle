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
#include "paddle/math/Matrix.h"
#include "paddle/math/BaseMatrix.h"

namespace paddle {

class PriorBoxLayer : public Layer {
public:
  explicit PriorBoxLayer(const LayerConfig& config) : Layer(config) {}
  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);
  void forward(PassType passType);
  void backward(const UpdateCallback& callback) {}
  int numPriors_;
  std::vector<int> minSize_;
  std::vector<int> maxSize_;
  std::vector<float> aspectRatio_;
  std::vector<float> variance_;
  MatrixPtr buffer_;
};

bool PriorBoxLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  auto pb_conf = config_.inputs(0).priorbox_conf();
  std::copy(pb_conf.min_size().begin(),
            pb_conf.min_size().end(),
            std::back_inserter(minSize_));
  std::copy(pb_conf.max_size().begin(),
            pb_conf.max_size().end(),
            std::back_inserter(maxSize_));
  std::copy(pb_conf.aspect_ratio().begin(),
            pb_conf.aspect_ratio().end(),
            std::back_inserter(aspectRatio_));
  std::copy(pb_conf.variance().begin(),
            pb_conf.variance().end(),
            std::back_inserter(variance_));
  // flip
  int input_ratio_length = aspectRatio_.size();
  for (int index = 0; index < input_ratio_length; index++)
    aspectRatio_.push_back(1 / aspectRatio_[index]);
  aspectRatio_.push_back(1.);
  numPriors_ = aspectRatio_.size();
  if (maxSize_.size() > 0) numPriors_++;
  buffer_ = Matrix::create(1, 1, false, false);
  return true;
}

void PriorBoxLayer::forward(PassType passType) {
  Layer::forward(passType);
  auto input = getInput(0);
  int layer_width = input.getFrameWidth();
  int layer_height = input.getFrameHeight();

  MatrixPtr inV1 = getInputValue(1);
  int image_width = inV1->getElement(0, 0);
  int image_height = inV1->getElement(0, 1);
  float step_w = static_cast<float>(image_width) / layer_width;
  float step_h = static_cast<float>(image_height) / layer_height;
  int dim = layer_height * layer_width * numPriors_ * 4;
  reserveOutput(1, dim * 2);
  // use a cpu buffer to compute
  Matrix::resizeOrCreate(buffer_, 1, dim * 2, false, false);
  auto* tmp_ptr = buffer_->getData();

  int idx = 0;
  for (int h = 0; h < layer_height; ++h) {
    for (int w = 0; w < layer_width; ++w) {
      float center_x = (w + 0.5) * step_w;
      float center_y = (h + 0.5) * step_h;
      int min_size = 0;
      for (size_t s = 0; s < minSize_.size(); s++) {
        // first prior.
        min_size = minSize_[s];
        int box_width = min_size;
        int box_height = min_size;
        // xmin, ymin, xmax, ymax.
        tmp_ptr[idx++] = (center_x - box_width / 2.) / image_width;
        tmp_ptr[idx++] = (center_y - box_height / 2.) / image_height;
        tmp_ptr[idx++] = (center_x + box_width / 2.) / image_width;
        tmp_ptr[idx++] = (center_y + box_height / 2.) / image_height;

        if (maxSize_.size() > 0) {
          CHECK_EQ(minSize_.size(), maxSize_.size());
          // second prior.
          for (size_t s = 0; s < maxSize_.size(); s++) {
            int max_size = maxSize_[s];
            box_width = box_height = sqrt(min_size * max_size);
            tmp_ptr[idx++] = (center_x - box_width / 2.) / image_width;
            tmp_ptr[idx++] = (center_y - box_height / 2.) / image_height;
            tmp_ptr[idx++] = (center_x + box_width / 2.) / image_width;
            tmp_ptr[idx++] = (center_y + box_height / 2.) / image_height;
          }
        }
      }
      // rest of priors.
      for (size_t r = 0; r < aspectRatio_.size(); r++) {
        float ar = aspectRatio_[r];
        if (fabs(ar - 1.) < 1e-6) continue;
        float box_width = min_size * sqrt(ar);
        float box_height = min_size / sqrt(ar);
        tmp_ptr[idx++] = (center_x - box_width / 2.) / image_width;
        tmp_ptr[idx++] = (center_y - box_height / 2.) / image_height;
        tmp_ptr[idx++] = (center_x + box_width / 2.) / image_width;
        tmp_ptr[idx++] = (center_y + box_height / 2.) / image_height;
      }
    }
  }
  // clip the prior's coordidate such that it is within [0, 1]
  for (int d = 0; d < dim; ++d)
    tmp_ptr[d] = std::min(std::max(tmp_ptr[d], (float)0.), (float)1.);
  // set the variance.
  for (int h = 0; h < layer_height; h++)
    for (int w = 0; w < layer_width; w++)
      for (int i = 0; i < numPriors_; i++)
        for (int j = 0; j < 4; j++) tmp_ptr[idx++] = variance_[j];
  MatrixPtr outV = getOutputValue();
  outV->copyFrom(buffer_->data_, dim * 2);
}
REGISTER_LAYER(priorbox, PriorBoxLayer);

}  // namespace paddle
