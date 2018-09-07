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
#include "LstmCompute.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/*
 * LstmStepLayer used in recurrent layer group.
 */
class LstmStepLayer : public Layer, public LstmCompute {
 protected:
  Argument state_;
  Argument gate_;
  Argument stateActive_;
  MatrixPtr checkIg_, checkFg_, checkOg_;
  MatrixPtr checkIgGrad_, checkFgGrad_, checkOgGrad_;
  std::unique_ptr<Weight> weight_;

 public:
  explicit LstmStepLayer(const LayerConfig& config) : Layer(config) {}

  ~LstmStepLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(lstm_step, LstmStepLayer);

bool LstmStepLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;
  CHECK_EQ(2U, inputLayers_.size());

  checkIg_ = Matrix::create(nullptr,
                            /* height= */ 1,
                            getSize(),
                            /* trans= */ false,
                            useGpu_);
  checkFg_ = Matrix::create(nullptr,
                            /* height= */ 1,
                            getSize(),
                            /* trans= */ false,
                            useGpu_);
  checkOg_ = Matrix::create(nullptr,
                            /* height= */ 1,
                            getSize(),
                            /* trans= */ false,
                            useGpu_);
  checkIgGrad_ = Matrix::create(nullptr,
                                /* height= */ 1,
                                getSize(),
                                /* trans= */ false,
                                useGpu_);
  checkFgGrad_ = Matrix::create(nullptr,
                                /* height= */ 1,
                                getSize(),
                                /* trans= */ false,
                                useGpu_);
  checkOgGrad_ = Matrix::create(nullptr,
                                /* height= */ 1,
                                getSize(),
                                /* trans= */ false,
                                useGpu_);

  if (biasParameter_.get() != NULL) {
    CHECK_EQ(getSize() * 3, biasParameter_->getSize());
    weight_.reset(new Weight(1, getSize() * 3, biasParameter_));
    if (weight_->getW()) {
      real* data = weight_->getW()->getData();
      checkIg_->setData(data);
      checkFg_->setData(data + getSize());
      checkOg_->setData(data + getSize() * 2);
    }

    if (weight_->getWGrad()) {
      real* data = weight_->getWGrad()->getData();
      checkIgGrad_->setData(data);
      checkFgGrad_->setData(data + getSize());
      checkOgGrad_->setData(data + getSize() * 2);
    }
  }

  setOutput("state", &state_);
  LstmCompute::init(config_);
  return true;
}

void LstmStepLayer::forward(PassType passType) {
  REGISTER_TIMER_INFO("LstmRecurrentFwTime", getName().c_str());
  Layer::forward(passType);

  const Argument& input = getInput(0);
  const Argument& prevState = getInput(1);
  CHECK_EQ(getSize() * 4, input.value->getWidth());
  CHECK_EQ(getSize(), prevState.value->getWidth());
  int batchSize = input.getBatchSize();
  reserveOutput(batchSize, getSize());
  resetSpecifyOutput(state_,
                     batchSize,
                     getSize(),
                     /*  isValueClean */ false,
                     /* isGradClean */ true);
  resetSpecifyOutput(gate_,
                     batchSize,
                     getSize() * 4,
                     /* isValueClean */ false,
                     /* isGradClean */ false);
  resetSpecifyOutput(stateActive_,
                     batchSize,
                     getSize(),
                     /*  isValueClean */ false,
                     /* isGradClean */ false);
  gate_.value->assign(*input.value);

  hl_lstm_value lstmValue;
  lstmValue.checkIg = checkIg_->getData();
  lstmValue.checkFg = checkFg_->getData();
  lstmValue.checkOg = checkOg_->getData();
  lstmValue.gateValue = gate_.value->getData();
  lstmValue.stateValue = state_.value->getData();
  lstmValue.prevStateValue = prevState.value->getData();
  lstmValue.stateActiveValue = stateActive_.value->getData();
  lstmValue.outputValue = output_.value->getData();

  if (useGpu_) {
    LstmCompute::forwardBatch<1>(lstmValue, getSize(), batchSize);
  } else {
    LstmCompute::forwardBatch<0>(lstmValue, getSize(), batchSize);
  }
}

void LstmStepLayer::backward(const UpdateCallback& callback) {
  REGISTER_TIMER_INFO("LstmRecurrentBwTime", getName().c_str());
  const Argument& input = getInput(0);
  const Argument& prevState = getInput(1);
  int batchSize = input.getBatchSize();

  hl_lstm_value lstmValue;
  hl_lstm_grad lstmGrad;
  lstmValue.checkIg = checkIg_->getData();
  lstmValue.checkFg = checkFg_->getData();
  lstmValue.checkOg = checkOg_->getData();
  lstmValue.gateValue = gate_.value->getData();
  lstmValue.prevStateValue = prevState.value->getData();
  lstmValue.stateValue = state_.value->getData();
  lstmValue.stateActiveValue = stateActive_.value->getData();

  lstmGrad.gateGrad = gate_.grad->getData();
  if (prevState.grad) {
    lstmGrad.prevStateGrad = prevState.grad->getData();
  } else {
    lstmGrad.prevStateGrad = nullptr;
  }
  lstmGrad.stateGrad = state_.grad->getData();
  lstmGrad.stateActiveGrad = stateActive_.grad->getData();
  lstmGrad.outputGrad = output_.grad->getData();
  lstmGrad.checkIgGrad = checkIgGrad_->getData();
  lstmGrad.checkFgGrad = checkFgGrad_->getData();
  lstmGrad.checkOgGrad = checkOgGrad_->getData();

  if (useGpu_) {
    LstmCompute::backwardBatch<1>(lstmValue, lstmGrad, getSize(), batchSize);
  } else {
    LstmCompute::backwardBatch<0>(lstmValue, lstmGrad, getSize(), batchSize);
  }

  if (input.grad) {
    input.grad->add(*gate_.grad);
  }

  if (weight_) {
    weight_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
