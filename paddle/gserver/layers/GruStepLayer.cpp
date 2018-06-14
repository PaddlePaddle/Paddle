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

#include "GruCompute.h"
#include "Layer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * @brief GruStepLayer is like GatedRecurrentLayer, but used in recurrent
 * layer group. GruStepLayer takes 2 input layer.
 * - input[0] with size * 3 and diveded into 3 equal parts: (xz_t, xr_t, xi_t).
 * - input[1] with size: {prev_out}.
 *
 * parameter and biasParameter is also diveded into 3 equal parts:
 * - parameter consists of (U_z, U_r, U)
 * - baisParameter consists of (bias_z, bias_r, bias_o)
 *
 * \f[
 * update \ gate: z_t = actGate(xz_t + U_z * prev_out + bias_z) \\
 * reset \ gate: r_t = actGate(xr_t + U_r * prev_out + bias_r)  \\
 * output \ candidate: {h}_t = actNode(xi_t + U * dot(r_t, prev_out) + bias_o)
 * \\
 * output: h_t = dot((1-z_t), prev_out) + dot(z_t, prev_out)
 * \f]
 *
 * @note
 *   - dot denotes "element-wise multiplication".
 *   - actNode is defined by config active_type
 *   - actGate is defined by config actvie_gate_type
 *
 * The config file api if gru_step_layer.
 */
class GruStepLayer : public Layer, public GruCompute {
 protected:
  Argument gate_;
  Argument resetOutput_;
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> bias_;

 public:
  explicit GruStepLayer(const LayerConfig& config) : Layer(config) {}

  ~GruStepLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(gru_step, GruStepLayer);

bool GruStepLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;
  CHECK_EQ(2U, inputLayers_.size());

  CHECK_EQ(getSize() * getSize() * 3, parameters_[0]->getSize());
  weight_.reset(new Weight(getSize(), getSize() * 3, parameters_[0]));

  if (biasParameter_.get() != NULL) {
    CHECK_EQ(getSize() * 3, biasParameter_->getSize());
    bias_.reset(new Weight(1, getSize() * 3, biasParameter_));
  }

  GruCompute::init(config_);
  return true;
}

void GruStepLayer::forward(PassType passType) {
  REGISTER_TIMER_INFO("GruStepFwTime", getName().c_str());
  Layer::forward(passType);

  const Argument& input = getInput(0);
  const Argument& prevOutput = getInput(1);
  CHECK_EQ(getSize() * 3, input.value->getWidth());
  CHECK_EQ(getSize(), prevOutput.value->getWidth());

  int batchSize = input.getBatchSize();
  resetOutput(batchSize, getSize());
  resetSpecifyOutput(gate_,
                     batchSize,
                     getSize() * 3,
                     /* isValueClean */ false,
                     /* isGradClean */ false);
  resetSpecifyOutput(resetOutput_,
                     batchSize,
                     getSize(),
                     /* isValueClean */ false,
                     /* isGradClean */ false);
  gate_.value->assign(*input.value);
  if (bias_) {
    gate_.value->addBias(*(bias_->getW()), 1);
  }

  hl_gru_value gruValue;
  gruValue.gateWeight = weight_->getW()->getData();
  gruValue.stateWeight = weight_->getW()->getData() + getSize() * getSize() * 2;
  gruValue.gateValue = gate_.value->getData();
  gruValue.resetOutputValue = resetOutput_.value->getData();
  gruValue.outputValue = output_.value->getData();
  gruValue.prevOutValue = prevOutput.value->getData();

  if (useGpu_) {
    GruCompute::forward<1>(gruValue, getSize(), batchSize);
  } else {
    GruCompute::forward<0>(gruValue, getSize(), batchSize);
  }
}

void GruStepLayer::backward(const UpdateCallback& callback) {
  REGISTER_TIMER_INFO("GruStepBwTime", getName().c_str());

  const Argument& input = getInput(0);
  const Argument& prevOutput = getInput(1);
  int batchSize = input.getBatchSize();

  hl_gru_value gruValue;
  gruValue.gateWeight = weight_->getW()->getData();
  gruValue.stateWeight = weight_->getW()->getData() + getSize() * getSize() * 2;
  gruValue.gateValue = gate_.value->getData();
  gruValue.resetOutputValue = resetOutput_.value->getData();
  gruValue.outputValue = output_.value->getData();
  gruValue.prevOutValue = prevOutput.value->getData();

  hl_gru_grad gruGrad;
  gruGrad.gateWeightGrad =
      (weight_->getWGrad() ? weight_->getWGrad()->getData() : nullptr);
  gruGrad.stateWeightGrad =
      (weight_->getWGrad()
           ? weight_->getWGrad()->getData() + getSize() * getSize() * 2
           : nullptr);

  gruGrad.gateGrad = gate_.grad->getData();
  gruGrad.resetOutputGrad = resetOutput_.grad->getData();
  gruGrad.outputGrad = output_.grad->getData();
  if (prevOutput.grad) {
    gruGrad.prevOutGrad = prevOutput.grad->getData();
  } else {
    gruGrad.prevOutGrad = nullptr;
  }

  if (useGpu_) {
    GruCompute::backward<1>(gruValue, gruGrad, getSize(), batchSize);
  } else {
    GruCompute::backward<0>(gruValue, gruGrad, getSize(), batchSize);
  }

  if (input.grad) {
    input.grad->add(*gate_.grad);
  }

  if (bias_ && bias_->getWGrad()) {
    bias_->getWGrad()->collectBias(*gate_.grad, 1);
  }

  if (bias_) {
    bias_->getParameterPtr()->incUpdate(callback);
  }
  weight_->getParameterPtr()->incUpdate(callback);
}

}  // namespace paddle
