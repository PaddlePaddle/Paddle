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

#include <memory>
#include <random>

#include "Layer.h"

namespace paddle {

/**
 * @brief A layer for sampling id from multinomial distribution from the
 * input layer. Sampling one id for one sample. The result is stored in
 * output_.ids.
 *
 * The config file api is sampling_id_layer.
 */
class SamplingIdLayer : public Layer {
  /// Produces random floating-point values, uniformly distributed on [0, 1).
  std::uniform_real_distribution<double> rand1_;
  std::vector<Argument> tmpCpuInput_;

 public:
  explicit SamplingIdLayer(const LayerConfig& config)
      : Layer(config), rand1_(0, 1) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override {
    bool ret = Layer::init(layerMap, parameterMap);
    CHECK_EQ(1UL, inputLayers_.size());
    if (useGpu_) {
      tmpCpuInput_.reserve(inputLayers_.size());
      for (size_t i = 0; i < inputLayers_.size(); i++) {
        tmpCpuInput_.push_back(Argument());
      }
    }
    return ret;
  }

  void forward(PassType passType) override {
    Layer::forward(passType);
    if (useGpu_) {
      for (size_t i = 0; i < inputLayers_.size(); i++) {
        tmpCpuInput_[i].resizeAndCopyFrom(
            getInput(i), false, HPPL_STREAM_DEFAULT);
      }
      hl_stream_synchronize(HPPL_STREAM_DEFAULT);
      forwardImp(tmpCpuInput_[0]);
    } else {
      forwardImp(getInput(0));
    }
  }

  void forwardImp(const Argument& input) {
    size_t batchSize = input.getBatchSize();
    IVector::resizeOrCreate(output_.ids, batchSize, useGpu_);
    real* buf = input.value->getData();
    int dim = input.value->getWidth();
    std::vector<int> ids(batchSize);
    auto& reng = ThreadLocalRandomEngine::get();
    for (size_t i = 0; i < batchSize; ++i) {
      double r = rand1_(reng);
      int id = dim - 1;
      for (int j = 0; j < dim; ++j) {
        if ((r -= buf[i * dim + j]) < 0) {
          id = j;
          break;
        }
      }
      ids[i] = id;
    }
    output_.ids->copyFrom(ids.data(), batchSize);
  }

  void backward(const UpdateCallback& callback) override {}
};

REGISTER_LAYER(sampling_id, SamplingIdLayer);

}  // namespace paddle
