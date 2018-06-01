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
#include "Projection.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * A concatenate layer has multiple input layers. It concatenates rows of
 * each input as one row for the output of this layer and apply activation.
 */
class ConcatenateLayer : public Layer {
 public:
  explicit ConcatenateLayer(const LayerConfig& config) : Layer(config) {}

  ~ConcatenateLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(concat, ConcatenateLayer);

bool ConcatenateLayer::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  if (!Layer::init(layerMap, parameterMap)) return false;

  CHECK(!biasParameter_);

  return true;
}

void ConcatenateLayer::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = getInput(0).getBatchSize();
  int size = getSize();
  reserveOutput(batchSize, size);

  const MatrixPtr& out = getOutputValue();
  int offset = 0;

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    const MatrixPtr& in = getInputValue(i);
    size_t inSize = in->getWidth();
    out->assignAtOffset(*in, offset);
    offset += inSize;
  }
  CHECK_EQ(size, offset);

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void ConcatenateLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  /* Do activation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  const MatrixPtr& out = getOutputGrad();
  int offset = 0;

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    const MatrixPtr& in = getInputGrad(i);
    size_t inSize = getInputValue(i)->getWidth();
    if (in) {
      in->addAtOffset(*out, offset);
    }
    offset += inSize;
  }
}

/**
 * concat2 layer is like concat layer, but each input layer was
 * processed by a Projection.
 */
class ConcatenateLayer2 : public Layer {
 public:
  explicit ConcatenateLayer2(const LayerConfig& config) : Layer(config) {}

  ~ConcatenateLayer2() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 protected:
  std::vector<std::unique_ptr<Projection>> projections_;
  std::vector<Argument> projOutput_;
  std::vector<std::pair<size_t, size_t>> projCol_;
  bool sharedBias_;
  std::unique_ptr<Weight> biases_;
};

REGISTER_LAYER(concat2, ConcatenateLayer2);

bool ConcatenateLayer2::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  if (!Layer::init(layerMap, parameterMap)) return false;

  CHECK_EQ(inputLayers_.size(), parameters_.size());
  projections_.reserve(inputLayers_.size());
  projCol_.reserve(inputLayers_.size());
  projOutput_.resize(inputLayers_.size());

  size_t startCol = 0;
  size_t endCol = 0;
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    projections_.emplace_back(Projection::create(
        config_.inputs(i).proj_conf(), parameters_[i], useGpu_));

    endCol += projections_[i]->getOutputSize();
    projCol_.push_back(std::make_pair(startCol, endCol));
    startCol = endCol;
  }
  CHECK_EQ(getSize(), endCol);

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    sharedBias_ = config_.shared_biases();
    size_t psize = config_.bias_size();
    biases_ = std::unique_ptr<Weight>(new Weight(1, psize, biasParameter_));
  }

  return true;
}

void ConcatenateLayer2::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = getInput(0).getBatchSize();
  int size = getSize();
  resetOutput(batchSize, size);

  for (size_t i = 0; i < projections_.size(); i++) {
    size_t startCol = projCol_[i].first;
    size_t endCol = projCol_[i].second;
    projOutput_[i].value = output_.value->subColMatrix(startCol, endCol);
    if (output_.grad) {
      projOutput_[i].grad = output_.grad->subColMatrix(startCol, endCol);
    }
  }

  {
    AsyncGpuBlock block;
    for (size_t i = 0; i != inputLayers_.size(); ++i) {
      projections_[i]->forward(&getInput(i), &projOutput_[i], passType);
    }
  }

  /* add the bias-vector */
  if (biases_) {
    REGISTER_TIMER_INFO("FwBiasTimer", getName().c_str());
    output_.value->addBias(*(biases_->getW()), 1, sharedBias_);
  }

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void ConcatenateLayer2::backward(const UpdateCallback& callback) {
  /* Do activation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  AsyncGpuBlock block;
  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("Concat2BpBiasTimer", getName().c_str());
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1, sharedBias_);
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    if (projections_[i]) {
      projections_[i]->backward(callback);
    }
  }
}

}  // namespace paddle
