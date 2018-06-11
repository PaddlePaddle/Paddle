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

namespace paddle {

/**
 * A layer for finding the id which has the maximal value for each sample.
 * The result is stored in output_.ids.
 *
 * The config file api is maxid_layer.
 */
class MaxIdLayer : public Layer {
 private:
  /// a predetermined number of best states at each level
  size_t beamSize_;

 public:
  explicit MaxIdLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override {
    bool ret = Layer::init(layerMap, parameterMap);
    CHECK_EQ(1UL, inputLayers_.size());

    beamSize_ = config_.has_beam_size() ? config_.beam_size() : FLAGS_beam_size;
    CHECK_GE(beamSize_, 1LU);
    return ret;
  }

  void forward(PassType passType) override {
    Layer::forward(passType);
    const Argument& input = getInput(0);
    size_t batchSize = input.getBatchSize();
    IVector::resizeOrCreate(output_.ids, batchSize * beamSize_, useGpu_);
    Matrix::resizeOrCreate(output_.in,
                           batchSize,
                           beamSize_,
                           false,
                           /* useGpu */ useGpu_);
    output_.value = nullptr;
    input.value->rowMax(*output_.ids, *output_.in);
  }

  void backward(const UpdateCallback& callback) override {}
};

REGISTER_LAYER(maxid, MaxIdLayer);

}  // namespace paddle
