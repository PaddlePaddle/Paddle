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

#include "GatedRecurrentLayer.h"
#include "Layer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(gated_recurrent, GatedRecurrentLayer);

bool GatedRecurrentLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;
  CHECK_EQ(1U, inputLayers_.size());
  CHECK_EQ(1U, parameters_.size());
  CHECK_EQ(getSize() * getSize() * 3, parameters_[0]->getSize());
  CHECK_EQ(getSize() * 3, biasParameter_->getSize());
  weight_.reset(new Weight(getSize(), getSize() * 3, parameters_[0]));
  gateWeight_.reset(new Weight(getSize(), getSize() * 2, parameters_[0], 0));
  stateWeight_.reset(new Weight(
      getSize(), getSize(), parameters_[0], 2 * getSize() * getSize()));
  if (biasParameter_.get() != NULL) {
    bias_.reset(new Weight(1, getSize() * 3, biasParameter_));
  }

  reversed_ = config_.reversed();
  activationGate_.reset(ActivationFunction::create(config_.active_gate_type()));

  GruCompute::init(config_);
  useBatch_ = true;

  return true;
}

void GatedRecurrentLayer::resetState() {
  CHECK(!reversed_) << "state is not allowed for reversed gated "
                       "recurrent layer";
  Matrix::resizeOrCreate(
      prevOutput_, 1, getSize(), /* trans= */ false, useGpu_);
  prevOutput_->zeroMem();

  // TODO(hedaoyuan): support prev_batch_state
  CHECK(!FLAGS_prev_batch_state) << "Not supported";

  useBatch_ = false;
}

void GatedRecurrentLayer::setState(LayerStatePtr state) {
  CHECK(state->value.size() == 1)
      << "one matrix is expected for GatedRecurrentLayer state";
  prevOutput_->copyFrom(*(state->value[0]));
}

LayerStatePtr GatedRecurrentLayer::getState() {
  LayerStatePtr res = std::make_shared<LayerState>();
  res->value.push_back(prevOutput_->clone(0, 0, useGpu_));
  res->value[0]->copyFrom(*prevOutput_);
  return res;
}

void GatedRecurrentLayer::forward(PassType passType) {
  REGISTER_TIMER_INFO("GruFwTimer", getName().c_str());
  Layer::forward(passType);

  const Argument& input = getInput(0);
  CHECK(input.sequenceStartPositions);
  int batchSize = input.getBatchSize();
  size_t numSequences = input.getNumSequences();
  resetOutput(batchSize, getSize());
  CHECK_EQ(getSize() * 3, input.value->getWidth());
  const int* starts = input.sequenceStartPositions->getData(false);
  // batchSize = length of total frames in a batch (NOT size of mini-batch)
  CHECK_EQ(starts[numSequences], batchSize);

  Matrix::resizeOrCreate(gate_.value,
                         /* height= */ batchSize,
                         getSize() * 3,
                         /* trans= */ false,
                         useGpu_);
  Matrix::resizeOrCreate(resetOutput_.value,
                         /* height= */ batchSize,
                         getSize(),
                         /* trans= */ false,
                         useGpu_);

  if (useBatch_) {
    forwardBatch(batchSize, numSequences, starts, input.value);
  } else {
    forwardSequence(batchSize, numSequences, starts, input.value);
  }
}

void GatedRecurrentLayer::backward(const UpdateCallback& callback) {
  REGISTER_TIMER_INFO("GruBwTimer", getName().c_str());
  const Argument& input = getInput(0);
  CHECK(input.sequenceStartPositions);
  int batchSize = input.getBatchSize();
  const int* starts = input.sequenceStartPositions->getData(false);
  size_t numSequences = input.getNumSequences();

  Matrix::resizeOrCreate(gate_.grad,
                         /* height= */ batchSize,
                         getSize() * 3,
                         /* trans= */ false,
                         useGpu_);
  Matrix::resizeOrCreate(resetOutput_.grad,
                         /* height= */ batchSize,
                         getSize(),
                         /* trans= */ false,
                         useGpu_);

  if (useBatch_) {
    backwardBatch(batchSize, input.grad);
  } else {
    backwardSequence(batchSize, numSequences, starts, input.grad);
  }

  if (bias_) {
    bias_->getParameterPtr()->incUpdate(callback);
  }

  weight_->getParameterPtr()->incUpdate(callback);
}

void GatedRecurrentLayer::forwardSequence(int batchSize,
                                          size_t numSequences,
                                          const int* starts,
                                          MatrixPtr inputValue) {
  REGISTER_TIMER_INFO("GruFwSequenceTime", getName().c_str());
  gate_.value->assign(*inputValue);
  if (bias_) {
    gate_.value->addBias(*(bias_->getW()), 1);
  }

  hl_gru_value gruValue;
  gruValue.gateWeight = (gateWeight_->getW())->getData();
  gruValue.stateWeight = (stateWeight_->getW())->getData();
  gruValue.gateValue = gate_.value->getData();
  gruValue.resetOutputValue = resetOutput_.value->getData();
  gruValue.outputValue = output_.value->getData();
  gruValue.prevOutValue = nullptr;

  if (reversed_) {
    gruValue.gateValue += (batchSize - 1) * getSize() * 3;
    gruValue.resetOutputValue += (batchSize - 1) * getSize();
    gruValue.outputValue += (batchSize - 1) * getSize();
  }

  auto nextFrame = [&gruValue](bool reversed, int frameSize) {
    gruValue.prevOutValue = gruValue.outputValue;
    if (!reversed) {
      gruValue.gateValue += frameSize * 3;
      gruValue.resetOutputValue += frameSize;
      gruValue.outputValue += frameSize;
    } else {
      gruValue.gateValue -= frameSize * 3;
      gruValue.resetOutputValue -= frameSize;
      gruValue.outputValue -= frameSize;
    }
  };

  if (!reversed_) {
    if (prevOutput_) {
      gruValue.prevOutValue = prevOutput_->getData();
    }
  }
  AsyncGpuBlock asyncGpuBlock;
  for (size_t n = 0; n < numSequences; ++n) {
    int length;
    if (!reversed_) {
      length = starts[n + 1] - starts[n];
    } else {
      length = starts[numSequences - n] - starts[numSequences - n - 1];
    }
    for (int l = 0; l < length; ++l) {
      if (useGpu_) {
        GruCompute::forward<1>(gruValue, getSize());
      } else {
        GruCompute::forward<0>(gruValue, getSize());
      }

      nextFrame(reversed_, getSize());
    }
    if (!reversed_) {
      if (!prevOutput_) gruValue.prevOutValue = nullptr;
    } else {
      gruValue.prevOutValue = nullptr;
    }
  }

  if (!reversed_) {
    if (prevOutput_) {
      prevOutput_->assign(*output_.value->subMatrix(batchSize - 1, 1));
    }
  }
}

void GatedRecurrentLayer::backwardSequence(int batchSize,
                                           size_t numSequences,
                                           const int* starts,
                                           MatrixPtr inputGrad) {
  REGISTER_TIMER_INFO("GruBwSequenceTime", getName().c_str());

  hl_gru_value gruValue;
  gruValue.gateWeight = (gateWeight_->getW())->getData();
  gruValue.stateWeight = (stateWeight_->getW())->getData();
  gruValue.gateValue = gate_.value->getData();
  gruValue.resetOutputValue = resetOutput_.value->getData();
  gruValue.outputValue = output_.value->getData();

  hl_gru_grad gruGrad;
  gruGrad.gateWeightGrad =
      (gateWeight_->getWGrad() ? gateWeight_->getWGrad()->getData() : nullptr);
  gruGrad.stateWeightGrad =
      (stateWeight_->getWGrad() ? stateWeight_->getWGrad()->getData()
                                : nullptr);
  gruGrad.gateGrad = gate_.grad->getData();
  gruGrad.resetOutputGrad = resetOutput_.grad->getData();
  gruGrad.outputGrad = output_.grad->getData();

  if (!reversed_) {
    gruValue.gateValue += (batchSize - 1) * getSize() * 3;
    gruValue.resetOutputValue += (batchSize - 1) * getSize();
    gruValue.outputValue += (batchSize - 1) * getSize();
    gruGrad.gateGrad += (batchSize - 1) * getSize() * 3;
    gruGrad.resetOutputGrad += (batchSize - 1) * getSize();
    gruGrad.outputGrad += (batchSize - 1) * getSize();
    gruValue.prevOutValue = gruValue.outputValue - getSize();
    gruGrad.prevOutGrad = gruGrad.outputGrad - getSize();
  } else {
    gruValue.prevOutValue = gruValue.outputValue + getSize();
    gruGrad.prevOutGrad = gruGrad.outputGrad + getSize();
  }

  auto nextFrame = [&gruValue, &gruGrad](bool reversed, int frameSize) {
    if (reversed) {
      gruValue.gateValue += frameSize * 3;
      gruValue.resetOutputValue += frameSize;
      gruValue.outputValue += frameSize;
      gruGrad.gateGrad += frameSize * 3;
      gruGrad.resetOutputGrad += frameSize;
      gruGrad.outputGrad += frameSize;
      gruValue.prevOutValue = gruValue.outputValue + frameSize;
      gruGrad.prevOutGrad = gruGrad.outputGrad + frameSize;
    } else {
      gruValue.gateValue -= frameSize * 3;
      gruValue.resetOutputValue -= frameSize;
      gruValue.outputValue -= frameSize;
      gruGrad.gateGrad -= frameSize * 3;
      gruGrad.resetOutputGrad -= frameSize;
      gruGrad.outputGrad -= frameSize;
      gruValue.prevOutValue = gruValue.outputValue - frameSize;
      gruGrad.prevOutGrad = gruGrad.outputGrad - frameSize;
    }
  };

  {
    AsyncGpuBlock asyncGpuBlock;
    for (size_t n = 0; n < numSequences; ++n) {
      int length;
      if (reversed_) {
        length = starts[n + 1] - starts[n];
      } else {
        length = starts[numSequences - n] - starts[numSequences - n - 1];
      }
      for (int l = 0; l < length; ++l) {
        if (l == length - 1) {
          gruValue.prevOutValue = nullptr;
          gruGrad.prevOutGrad = nullptr;
        }
        if (useGpu_) {
          GruCompute::backward<1>(gruValue, gruGrad, getSize());
        } else {
          GruCompute::backward<0>(gruValue, gruGrad, getSize());
        }
        nextFrame(reversed_, getSize());
      }
    }
  }

  if (inputGrad) {
    inputGrad->add(*gate_.grad);
  }
  if (bias_ && bias_->getWGrad()) {
    bias_->getWGrad()->collectBias(*gate_.grad, 1);
  }
}

void GatedRecurrentLayer::forwardBatch(int batchSize,
                                       size_t numSequences,
                                       const int* starts,
                                       MatrixPtr inputValue) {
  REGISTER_TIMER_INFO("GruFwBatchTime", getName().c_str());
  hl_gru_value gruValue;
  gruValue.gateWeight = (gateWeight_->getW())->getData();
  gruValue.stateWeight = (stateWeight_->getW())->getData();

  if (!batchValue_) {
    batchValue_.reset(new SequenceToBatch(useGpu_));
  }
  batchValue_->resizeOrCreateBatch(batchSize, numSequences, starts, reversed_);

  batchValue_->resizeOrCreate(*output_.value);
  batchValue_->copy(*inputValue, *gate_.value, /* seq2batch */ true);
  if (bias_) {
    gate_.value->addBias(*(bias_->getW()), 1);
  }

  {
    int numBatch = batchValue_->getNumBatch();
    int curBatchSize = 0;
    AsyncGpuBlock asyncGpuBlock;
    for (int n = 0; n < numBatch; n++) {
      MatrixPtr outputValueTmp = batchValue_->getBatchValue(n);
      gruValue.outputValue = outputValueTmp->getData();
      gruValue.gateValue =
          (batchValue_->getBatchValue(*gate_.value, n))->getData();
      gruValue.resetOutputValue =
          (batchValue_->getBatchValue(*resetOutput_.value, n))->getData();

      curBatchSize = outputValueTmp->getHeight();
      gruValue.prevOutValue =
          (n == 0
               ? nullptr
               : (batchValue_->getBatchValue(n - 1, curBatchSize))->getData());

      {
        if (useGpu_) {
          GruCompute::forward<1>(gruValue, getSize(), curBatchSize);
        } else {
          GruCompute::forward<0>(gruValue, getSize(), curBatchSize);
        }
      }
    }
  }
  { batchValue_->copyBackSeq(*output_.value); }
}

void GatedRecurrentLayer::backwardBatch(int batchSize, MatrixPtr inputGrad) {
  REGISTER_TIMER_INFO("GruBwBatchTime", getName().c_str());
  hl_gru_value gruValue;
  gruValue.gateWeight = (gateWeight_->getW())->getData();
  gruValue.stateWeight = (stateWeight_->getW())->getData();

  hl_gru_grad gruGrad;
  gruGrad.gateWeightGrad =
      (gateWeight_->getWGrad() ? gateWeight_->getWGrad()->getData() : nullptr);
  gruGrad.stateWeightGrad =
      (stateWeight_->getWGrad() ? stateWeight_->getWGrad()->getData()
                                : nullptr);

  if (!batchGrad_) {
    batchGrad_.reset(new SequenceToBatch(useGpu_));
  }
  batchGrad_->shareIndexWith(*batchValue_);

  { batchGrad_->copyFromSeq(*output_.grad); }

  {
    int numBatch = batchGrad_->getNumBatch();
    int batchSize = 0;
    AsyncGpuBlock asyncGpuBlock;
    for (int n = (int)numBatch - 1; n >= 0; n--) {
      gruValue.gateValue =
          (batchGrad_->getBatchValue(*gate_.value, n))->getData();
      gruValue.resetOutputValue =
          (batchGrad_->getBatchValue(*resetOutput_.value, n))->getData();

      MatrixPtr outputGradTmp = batchGrad_->getBatchValue(n);
      gruGrad.outputGrad = outputGradTmp->getData();
      gruGrad.gateGrad = (batchGrad_->getBatchValue(*gate_.grad, n))->getData();
      gruGrad.resetOutputGrad =
          (batchGrad_->getBatchValue(*resetOutput_.grad, n))->getData();

      {
        batchSize = outputGradTmp->getHeight();
        gruValue.prevOutValue =
            (n == 0
                 ? nullptr
                 : (batchValue_->getBatchValue(n - 1, batchSize))->getData());
        gruGrad.prevOutGrad =
            (n == 0 ? nullptr
                    : (batchGrad_->getBatchValue(n - 1, batchSize))->getData());

        if (useGpu_) {
          GruCompute::backward<1>(gruValue, gruGrad, getSize(), batchSize);
        } else {
          GruCompute::backward<0>(gruValue, gruGrad, getSize(), batchSize);
        }
      }
    }
  }

  if (inputGrad) {
    batchGrad_->add(*inputGrad, *gate_.grad, /* seq2batch */ false);
  }
  if (bias_ && bias_->getWGrad()) {
    bias_->getWGrad()->collectBias(*gate_.grad, /* scale */ 1);
  }
}

}  // namespace paddle
