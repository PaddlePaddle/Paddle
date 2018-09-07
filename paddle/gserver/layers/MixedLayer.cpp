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

#include "MixedLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(mixed, MixedLayer);

bool MixedLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  if (!Layer::init(layerMap, parameterMap)) return false;

  CHECK_EQ(inputLayers_.size(), parameters_.size());
  projections_.resize(inputLayers_.size());
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    if (config_.inputs(i).has_proj_conf()) {
      projections_[i].reset(Projection::create(
          config_.inputs(i).proj_conf(), parameters_[i], useGpu_));
    } else {
      CHECK(!parameters_[i]) << "should no parameters for operators";
    }
  }
  for (auto& operator_conf : config_.operator_confs()) {
    for (auto& input_index : operator_conf.input_indices()) {
      CHECK(!config_.inputs(input_index).has_proj_conf());
    }
    operators_.emplace_back(Operator::create(operator_conf, useGpu_));
  }

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    sharedBias_ = config_.shared_biases();
    size_t psize = config_.bias_size();
    biases_ = std::unique_ptr<Weight>(new Weight(1, psize, biasParameter_));
  }

  return true;
}

void MixedLayer::prefetch() {
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    if (projections_[i]) {
      projections_[i]->prefetch(&getInput(i));
    }
  }
}

void MixedLayer::resetState() {
  for (auto& proj : projections_) {
    if (proj) {
      proj->resetState();
    }
  }
}

void MixedLayer::setState(LayerStatePtr state) {
  CHECK(projectionStateMatrixSize_.size() == projections_.size())
      << "projection size mis-match";

  int start = 0;
  LayerStatePtr statePtr = std::make_shared<LayerState>();
  for (int i = 0; i < (int)projectionStateMatrixSize_.size(); i++) {
    if (projectionStateMatrixSize_[i] > 0) {
      statePtr->value.clear();
      for (int j = start; j < start + projectionStateMatrixSize_[i]; j++) {
        statePtr->value.push_back(state->value[j]);
      }
      projections_[i]->setState(statePtr);
      start += projectionStateMatrixSize_[i];
    }
  }
  CHECK((int)state->value.size() == start) << "state matrix size mis-match";
}

// Return state which consists of all projections states
LayerStatePtr MixedLayer::getState() {
  bool init = projectionStateMatrixSize_.size() == 0;
  LayerStatePtr res = std::make_shared<LayerState>();
  for (int i = 0; i < (int)projections_.size(); i++) {
    LayerStatePtr statePtr =
        projections_[i] ? projections_[i]->getState() : nullptr;
    int stateSize = statePtr == nullptr ? 0 : statePtr->value.size();
    if (init) {
      projectionStateMatrixSize_.push_back(stateSize);
    } else {
      CHECK(projectionStateMatrixSize_[i] == stateSize)
          << "state matrix size mis-match";
    }
    if (statePtr != nullptr) {
      for (auto& matrixPtr : statePtr->value) {
        res->value.push_back(matrixPtr);
      }
    }
  }
  return res;
}

void MixedLayer::forward(PassType passType) {
  Layer::forward(passType);

  int batchSize = getInput(0).getBatchSize();
  int size = getSize();
  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    resetOutput(batchSize, size);
  }

  MatrixPtr outV = getOutputValue();

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    if (projections_[i]) {
      projections_[i]->forward(&getInput(i), &output_, passType);
    }
  }

  std::vector<const Argument*> ins;
  for (auto& op : operators_) {
    ins.clear();
    for (auto& input_index : op->getConfig().input_indices()) {
      ins.push_back(&getInput(input_index));
    }
    op->forward(ins, &output_, passType);
  }

  /* add the bias-vector */
  if (biases_.get() != NULL) {
    REGISTER_TIMER_INFO("FwBiasTimer", getName().c_str());
    outV->addBias(*(biases_->getW()), 1, sharedBias_);
  }

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void MixedLayer::backward(const UpdateCallback& callback) {
  /* Do activation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  if (biases_ && biases_->getWGrad()) {
    REGISTER_TIMER_INFO("BpBiasTimer", getName().c_str());
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1, sharedBias_);

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    if (projections_[i]) {
      projections_[i]->backward(callback);
    }
  }

  for (auto& op : operators_) {
    op->backward();
  }
}

}  // namespace paddle
