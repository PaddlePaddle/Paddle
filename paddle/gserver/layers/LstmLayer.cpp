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

#include "LstmLayer.h"
#include "paddle/math/BaseMatrix.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Stat.h"

DECLARE_bool(prev_batch_state);

namespace paddle {

REGISTER_LAYER(lstmemory, LstmLayer);

bool LstmLayer::init(const LayerMap &layerMap,
                     const ParameterMap &parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;
  CHECK_EQ(1U, inputLayers_.size());
  CHECK_EQ(1U, parameters_.size());
  CHECK_EQ(getSize() * getSize() * 4, parameters_[0]->getSize());
  CHECK_EQ(getSize() * 7, biasParameter_->getSize());
  weight_.reset(new Weight(getSize(), getSize() * 4, parameters_[0]));
  if (biasParameter_.get() != NULL) {
    bias_.reset(new Weight(1, getSize() * 7, biasParameter_));
    if (bias_->getW()) {
      localBias_ = Matrix::create(nullptr,
                                  /* height= */ 1,
                                  getSize() * 4,
                                  /* trans= */ false,
                                  useGpu_);
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

      localBias_->setData(bias_->getW()->getData());
      checkIg_->setData(bias_->getW()->getData() + getSize() * 4);
      checkFg_->setData(bias_->getW()->getData() + getSize() * 5);
      checkOg_->setData(bias_->getW()->getData() + getSize() * 6);
    }

    if (bias_->getWGrad()) {
      localBiasGrad_ = Matrix::create(nullptr,
                                      /* height= */ 1,
                                      getSize() * 4,
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
      localBiasGrad_->setData(bias_->getWGrad()->getData());
      checkIgGrad_->setData(bias_->getWGrad()->getData() + getSize() * 4);
      checkFgGrad_->setData(bias_->getWGrad()->getData() + getSize() * 5);
      checkOgGrad_->setData(bias_->getWGrad()->getData() + getSize() * 6);
    }
  } else {
    LOG(FATAL) << "Bias should be here.";
  }
  reversed_ = config_.reversed();

  // create IdentityActivation for using drop_rate
  activation_.reset(ActivationFunction::create(""));

  LstmCompute::init(config_);
  useBatch_ = true;
  useSeqParallel_ = false;
  if (useGpu_ && (getSize() == 32 || getSize() == 64)) {
    useSeqParallel_ = true;
  }

  return true;
}

void LstmLayer::resetState() {
  CHECK(!reversed_) << "state is not allowed for reversed lstmemory layer";
  Matrix::resizeOrCreate(
      prevOutput_, 1, getSize(), /* trans= */ false, useGpu_);
  Matrix::resizeOrCreate(prevState_, 1, getSize(), /* trans= */ false, useGpu_);
  prevOutput_->resize(0, getSize());
  prevState_->resize(0, getSize());
  if (FLAGS_prev_batch_state) {
    useBatch_ = true;
  } else {
    useBatch_ = false;
  }
}

void LstmLayer::setState(LayerStatePtr state) {
  CHECK(state->value.size() == 2) << "two matrices are expected for LSTM state";
  prevOutput_->resize(state->value[0]->getHeight(),
                      state->value[0]->getWidth());
  prevState_->resize(state->value[1]->getHeight(), state->value[1]->getWidth());
  prevOutput_->copyFrom(*(state->value[0]));
  prevState_->copyFrom(*(state->value[1]));
}

LayerStatePtr LstmLayer::getState() {
  LayerStatePtr res = std::make_shared<LayerState>();
  if (prevOutput_->getHeight() && prevOutput_->getWidth()) {
    res->value.push_back(prevOutput_->clone(0, 0, useGpu_));
    res->value[0]->copyFrom(*prevOutput_);
    res->value.push_back(prevState_->clone(0, 0, useGpu_));
    res->value[1]->copyFrom(*prevState_);
  } else {
    MatrixPtr output =
        Matrix::create(1, getSize(), /* trans= */ false, useGpu_);
    MatrixPtr state = Matrix::create(1, getSize(), /* trans= */ false, useGpu_);
    output->resize(0, getSize());
    state->resize(0, getSize());
    res->value.push_back(output);
    res->value.push_back(state);
  }
  return res;
}

void LstmLayer::forward(PassType passType) {
  REGISTER_TIMER_INFO("LstmFwTimer", getName().c_str());
  Layer::forward(passType);

  const Argument &input = getInput(0);
  CHECK(input.sequenceStartPositions);
  int batchSize = input.getBatchSize();
  resetOutput(batchSize, getSize());
  CHECK_EQ(getSize() * 4, input.value->getWidth());
  size_t numSequences = input.getNumSequences();
  const int *starts = input.sequenceStartPositions->getData(false);
  CHECK_EQ(starts[numSequences], batchSize);

  Matrix::resizeOrCreate(gate_.value,
                         /* height= */ batchSize,
                         getSize() * 4,
                         /* trans= */ false,
                         useGpu_);
  if (prevOutput_) {
    size_t prevNumSeq = useBatch_ ? numSequences : 1;
    if (prevOutput_->getHeight() == 0) {
      prevOutput_->resize(prevNumSeq, getSize());
      prevState_->resize(prevNumSeq, getSize());
      prevOutput_->zeroMem();
      prevState_->zeroMem();
    } else {
      CHECK_EQ(prevOutput_->getHeight(), prevNumSeq)
          << "the number of sequences must be the same";
    }
    Matrix::resizeOrCreate(totalState_,
                           prevState_->getHeight() + batchSize,
                           getSize(),
                           /*trans*/ false,
                           useGpu_);
    state_.value = Matrix::create(nullptr,
                                  /* height= */ batchSize,
                                  getSize(),
                                  /* trans= */ false,
                                  useGpu_);
    state_.value->setData(totalState_->getData() +
                          prevState_->getHeight() * getSize());
  } else {
    Matrix::resizeOrCreate(state_.value,
                           /* height= */ batchSize,
                           getSize(),
                           /* trans= */ false,
                           useGpu_);
  }
  Matrix::resizeOrCreate(preOutput_.value,
                         /* height= */ batchSize,
                         getSize(),
                         /* trans= */ false,
                         useGpu_);

  if (!useBatch_) {
    forwardSequence(batchSize, numSequences, starts, input.value);
  } else {
    if (!useSeqParallel_) {
      forwardBatch(batchSize, numSequences, starts, input.value);
    } else {
      const int *starts = input.sequenceStartPositions->getData(useGpu_);
      forwardSeqParallel(batchSize, numSequences, starts, input.value);
    }
  }
  /*  activation */ { forwardActivation(); }
}

void LstmLayer::backward(const UpdateCallback &callback) {
  REGISTER_TIMER_INFO("LstmBwTimer", getName().c_str());
  /*  Do derivation */ { backwardActivation(); }

  const Argument &input = getInput(0);
  CHECK(input.sequenceStartPositions);
  int batchSize = input.getBatchSize();
  size_t numSequences = input.getNumSequences();

  Matrix::resizeOrCreate(gate_.grad,
                         /* height= */ batchSize,
                         getSize() * 4,
                         /* trans= */ false,
                         useGpu_);
  Matrix::resizeOrCreate(state_.grad,
                         /* height= */ batchSize,
                         getSize(),
                         /* trans= */ false,
                         useGpu_);
  Matrix::resizeOrCreate(preOutput_.grad,
                         /* height= */ batchSize,
                         getSize(),
                         /* trans= */ false,
                         useGpu_);
  state_.grad->zero();

  const int *starts = input.sequenceStartPositions->getData(false);
  if (!useBatch_) {
    backwardSequence(batchSize, numSequences, starts, input.grad);
  } else {
    if (!useSeqParallel_) {
      backwardBatch(batchSize, numSequences, starts, input.grad);
    } else {
      const int *starts = input.sequenceStartPositions->getData(useGpu_);
      backwardSeqParallel(batchSize, numSequences, starts, input.grad);
    }
  }

  if (bias_) {
    bias_->getParameterPtr()->incUpdate(callback);
  }
  weight_->getParameterPtr()->incUpdate(callback);
}

void LstmLayer::forwardSequence(int batchSize,
                                size_t numSequences,
                                const int *starts,
                                MatrixPtr inputValue) {
  REGISTER_TIMER_INFO("LstmFwSequenceTime", getName().c_str());
  gate_.value->assign(*inputValue);
  if (bias_) {
    gate_.value->addBias(*localBias_, 1);
  }

  hl_lstm_value lstmValue;
  lstmValue.checkIg = checkIg_->getData();
  lstmValue.checkFg = checkFg_->getData();
  lstmValue.checkOg = checkOg_->getData();
  lstmValue.gateValue = gate_.value->getData();
  lstmValue.stateValue = state_.value->getData();
  lstmValue.stateActiveValue = preOutput_.value->getData();
  lstmValue.outputValue = output_.value->getData();
  lstmValue.prevStateValue = nullptr;
  if (reversed_) {
    lstmValue.gateValue += (batchSize - 1) * getSize() * 4;
    lstmValue.stateValue += (batchSize - 1) * getSize();
    lstmValue.stateActiveValue += (batchSize - 1) * getSize();
    lstmValue.outputValue += (batchSize - 1) * getSize();
  }

  auto nextFrame = [&lstmValue](bool reversed, int frameSize) {
    lstmValue.prevStateValue = lstmValue.stateValue;
    if (!reversed) {
      lstmValue.gateValue += frameSize * 4;
      lstmValue.stateValue += frameSize;
      lstmValue.stateActiveValue += frameSize;
      lstmValue.outputValue += frameSize;
    } else {
      lstmValue.gateValue -= frameSize * 4;
      lstmValue.stateValue -= frameSize;
      lstmValue.stateActiveValue -= frameSize;
      lstmValue.outputValue -= frameSize;
    }
  };

  MatrixPtr frameGate = Matrix::create(nullptr,
                                       /* height= */ 1,
                                       getSize() * 4,
                                       /* trans= */ false,
                                       useGpu_);
  MatrixPtr frameOutput = Matrix::create(nullptr,
                                         /* height= */ 1,
                                         getSize(),
                                         /* trans= */ false,
                                         useGpu_);

  if (!reversed_) {
    if (prevState_) {
      lstmValue.prevStateValue = prevState_->getData();
    }
    if (prevOutput_) {
      frameGate->setData(lstmValue.gateValue);
      frameGate->mul(*prevOutput_, *weight_->getW(), 1, 1);
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
        LstmCompute::forwardOneSequence<1>(lstmValue, getSize());
      } else {
        LstmCompute::forwardOneSequence<0>(lstmValue, getSize());
      }

      if (l != length - 1) {
        frameOutput->setData(lstmValue.outputValue);
        nextFrame(reversed_, getSize());
        frameGate->setData(lstmValue.gateValue);
        frameGate->mul(*frameOutput, *weight_->getW(), 1, 1);
      }
    }
    if (n != numSequences - 1) {
      frameOutput->setData(lstmValue.outputValue);
      nextFrame(reversed_, getSize());
      frameGate->setData(lstmValue.gateValue);
      if (!reversed_) {
        if (!prevState_) lstmValue.prevStateValue = nullptr;
        if (prevOutput_) {
          frameGate->mul(*frameOutput, *weight_->getW(), 1, 1);
        }
      } else {
        lstmValue.prevStateValue = nullptr;
      }
    }
  }

  if (!reversed_) {
    if (prevState_) {
      prevState_->assign(*state_.value->subMatrix(batchSize - 1, 1));
    }
    if (prevOutput_) {
      prevOutput_->assign(*output_.value->subMatrix(batchSize - 1, 1));
    }
  }
}

void LstmLayer::backwardSequence(int batchSize,
                                 size_t numSequences,
                                 const int *starts,
                                 MatrixPtr inputGrad) {
  REGISTER_TIMER_INFO("LstmBwSequenceTime", getName().c_str());
  MatrixPtr weightT = weight_->getW()->getTranspose();

  hl_lstm_value lstmValue;
  hl_lstm_grad lstmGrad;
  lstmValue.checkIg = checkIg_->getData();
  lstmValue.checkFg = checkFg_->getData();
  lstmValue.checkOg = checkOg_->getData();
  lstmValue.gateValue = gate_.value->getData();
  lstmValue.stateValue = state_.value->getData();
  lstmValue.stateActiveValue = preOutput_.value->getData();
  lstmValue.outputValue = nullptr;

  if (bias_->getWGrad()) {
    lstmGrad.checkIgGrad = checkIgGrad_->getData();
    lstmGrad.checkFgGrad = checkFgGrad_->getData();
    lstmGrad.checkOgGrad = checkOgGrad_->getData();
  } else {
    lstmGrad.checkIgGrad = nullptr;
    lstmGrad.checkFgGrad = nullptr;
    lstmGrad.checkOgGrad = nullptr;
  }
  lstmGrad.gateGrad = gate_.grad->getData();
  lstmGrad.stateGrad = state_.grad->getData();
  lstmGrad.stateActiveGrad = nullptr;
  lstmGrad.outputGrad = output_.grad->getData();

  if (!reversed_) {
    lstmValue.gateValue += (batchSize - 1) * getSize() * 4;
    lstmGrad.gateGrad += (batchSize - 1) * getSize() * 4;
    lstmValue.stateValue += (batchSize - 1) * getSize();
    lstmGrad.stateGrad += (batchSize - 1) * getSize();
    lstmValue.stateActiveValue += (batchSize - 1) * getSize();
    lstmGrad.outputGrad += (batchSize - 1) * getSize();
    lstmValue.prevStateValue = lstmValue.stateValue - getSize();
    lstmGrad.prevStateGrad = lstmGrad.stateGrad - getSize();
  } else {
    lstmValue.prevStateValue = lstmValue.stateValue + getSize();
    lstmGrad.prevStateGrad = lstmGrad.stateGrad + getSize();
  }

  auto nextFrame = [&lstmValue, &lstmGrad](bool reversed, int frameSize) {
    if (reversed) {
      lstmValue.gateValue += frameSize * 4;
      lstmGrad.gateGrad += frameSize * 4;
      lstmValue.stateValue += frameSize;
      lstmGrad.stateGrad += frameSize;
      lstmValue.stateActiveValue += frameSize;
      lstmGrad.outputGrad += frameSize;
      lstmValue.prevStateValue = lstmValue.stateValue + frameSize;
      lstmGrad.prevStateGrad = lstmGrad.stateGrad + frameSize;
    } else {
      lstmValue.gateValue -= frameSize * 4;
      lstmGrad.gateGrad -= frameSize * 4;
      lstmValue.stateValue -= frameSize;
      lstmGrad.stateGrad -= frameSize;
      lstmValue.stateActiveValue -= frameSize;
      lstmGrad.outputGrad -= frameSize;
      lstmValue.prevStateValue = lstmValue.stateValue - frameSize;
      lstmGrad.prevStateGrad = lstmGrad.stateGrad - frameSize;
    }
  };

  MatrixPtr frameGate = Matrix::create(nullptr,
                                       /* height= */ 1,
                                       getSize() * 4,
                                       /* trans= */ false,
                                       useGpu_);
  MatrixPtr frameOutput = Matrix::create(nullptr,
                                         /* height= */ 1,
                                         getSize(),
                                         /* trans= */ false,
                                         useGpu_);

  {
    AsyncGpuBlock asyncGpuBlock;
    for (size_t n = 0; n < numSequences; ++n) {
      int length;
      int start;
      if (reversed_) {
        length = starts[n + 1] - starts[n];
        start = starts[n];
      } else {
        length = starts[numSequences - n] - starts[numSequences - n - 1];
        start = starts[numSequences - n - 1];
      }
      for (int l = 0; l < length; ++l) {
        if (l == length - 1) {
          lstmValue.prevStateValue = nullptr;
          lstmGrad.prevStateGrad = nullptr;
        }
        if (useGpu_) {
          LstmCompute::backwardOneSequence<1>(lstmValue, lstmGrad, getSize());
        } else {
          LstmCompute::backwardOneSequence<0>(lstmValue, lstmGrad, getSize());
        }
        if (l != length - 1) {
          frameGate->setData(lstmGrad.gateGrad);
          nextFrame(reversed_, getSize());
          frameOutput->setData(lstmGrad.outputGrad);
          frameOutput->mul(*frameGate, *weightT, 1, 1);
        } else {
          nextFrame(reversed_, getSize());
        }
      }

      if (weight_->getWGrad()) {
        if (!reversed_) {
          weight_->getWGrad()->mul(
              *output_.value->subMatrix(start, length - 1)->getTranspose(),
              *gate_.grad->subMatrix(start + 1, length - 1),
              1,
              1);
        } else {
          weight_->getWGrad()->mul(
              *output_.value->subMatrix(start + 1, length - 1)->getTranspose(),
              *gate_.grad->subMatrix(start, length - 1),
              1,
              1);
        }
      }
    }
  }

  if (inputGrad) {
    inputGrad->add(*gate_.grad);
  }
  if (bias_ && bias_->getWGrad()) {
    localBiasGrad_->collectBias(*gate_.grad, 1);
  }
}

void LstmLayer::forwardBatch(int batchSize,
                             size_t numSequences,
                             const int *starts,
                             MatrixPtr inputValue) {
  REGISTER_TIMER_INFO("LstmFwBatchTime", getName().c_str());

  hl_lstm_value lstmValue;
  lstmValue.checkIg = checkIg_->getData();
  lstmValue.checkFg = checkFg_->getData();
  lstmValue.checkOg = checkOg_->getData();

  if (!batchValue_) {
    batchValue_.reset(new SequenceToBatch(useGpu_));
  }
  batchValue_->resizeOrCreateBatch(
      batchSize, numSequences, starts, reversed_, prevOutput_ ? true : false);

  batchValue_->resizeOrCreate(*output_.value);
  batchValue_->copy(*inputValue, *gate_.value, /* seq2batch */ true);
  if (bias_) {
    gate_.value->addBias(*localBias_, 1);
  }

  {
    int numBatch = batchValue_->getNumBatch();
    int batchSize = 0;
    AsyncGpuBlock asyncGpuBlock;
    if (prevState_) {
      lstmValue.prevStateValue = totalState_->getData();
    } else {
      lstmValue.prevStateValue = nullptr;
    }
    for (int n = 0; n < numBatch; n++) {
      MatrixPtr outputValue = batchValue_->getBatchValue(n);
      MatrixPtr gateValue = batchValue_->getBatchValue(*gate_.value, n);
      batchSize = outputValue->getHeight();

      if (n != 0) {
        MatrixPtr batch1 = batchValue_->getBatchValue(n - 1, batchSize);
        gateValue->mul(*batch1, *weight_->getW(), 1, 1);
      } else if (prevOutput_) {
        Matrix::resizeOrCreate(prevBatchOutput2_,
                               gateValue->getHeight(),
                               getSize(),
                               false,
                               useGpu_);
        batchValue_->prevOutput2Batch(*prevOutput_, *prevBatchOutput2_);
        gateValue->mul(*prevBatchOutput2_, *weight_->getW(), 1, 1);

        batchValue_->prevOutput2Batch(*prevState_,
                                      *totalState_->subMatrix(0, numSequences));
      }

      lstmValue.gateValue = gateValue->getData();
      lstmValue.outputValue = outputValue->getData();
      lstmValue.stateValue =
          batchValue_->getBatchValue(*state_.value, n)->getData();
      lstmValue.stateActiveValue =
          batchValue_->getBatchValue(*preOutput_.value, n)->getData();
      {
        if (useGpu_) {
          LstmCompute::forwardBatch<1>(lstmValue, getSize(), batchSize);
        } else {
          LstmCompute::forwardBatch<0>(lstmValue, getSize(), batchSize);
        }
      }
      lstmValue.prevStateValue = lstmValue.stateValue;
    }
  }
  {
    REGISTER_TIMER_INFO("batchToSeq", getName().c_str());
    batchValue_->copyBackSeq(*output_.value);
  }
  if (prevOutput_) {
    getPrevBatchOutput(numSequences);
    getPrevBatchState(numSequences);
  }
}

void LstmLayer::getPrevBatchOutput(size_t numSequences) {
  prevOutput_->resize(numSequences, getSize());
  batchValue_->getSeqOutputFromBatch(*prevOutput_,
                                     *batchValue_->getBatchValue());
}

void LstmLayer::getPrevBatchState(size_t numSequences) {
  prevState_->resize(numSequences, getSize());
  batchValue_->getSeqOutputFromBatch(*prevState_, *state_.value);
}

void LstmLayer::backwardBatch(int batchSize,
                              size_t numSequences,
                              const int *starts,
                              MatrixPtr inputGrad) {
  REGISTER_TIMER_INFO("LstmBwBatchTime", getName().c_str());

  hl_lstm_value lstmValue;
  lstmValue.checkIg = checkIg_->getData();
  lstmValue.checkFg = checkFg_->getData();
  lstmValue.checkOg = checkOg_->getData();

  hl_lstm_grad lstmGrad;
  lstmGrad.stateActiveGrad = preOutput_.grad->getData();

  if (bias_->getWGrad()) {
    lstmGrad.checkIgGrad = checkIgGrad_->getData();
    lstmGrad.checkFgGrad = checkFgGrad_->getData();
    lstmGrad.checkOgGrad = checkOgGrad_->getData();
  } else {
    lstmGrad.checkIgGrad = nullptr;
    lstmGrad.checkFgGrad = nullptr;
    lstmGrad.checkOgGrad = nullptr;
  }

  if (!batchGrad_) {
    batchGrad_.reset(new SequenceToBatch(useGpu_));
  }
  batchGrad_->shareIndexWith(*batchValue_);

  {
    REGISTER_TIMER_INFO("seqToBatch", getName().c_str());
    batchGrad_->copyFromSeq(*output_.grad);
  }

  {
    MatrixPtr weightT = weight_->getW()->getTranspose();
    int numBatch = batchGrad_->getNumBatch();
    int batchSize = 0;
    AsyncGpuBlock asyncGpuBlock;
    for (int n = (int)numBatch - 1; n >= 0; n--) {
      MatrixPtr outputGrad = batchGrad_->getBatchValue(n);
      MatrixPtr gateGrad = batchGrad_->getBatchValue(*gate_.grad, n);

      lstmValue.gateValue =
          batchGrad_->getBatchValue(*gate_.value, n)->getData();
      lstmValue.stateValue =
          batchGrad_->getBatchValue(*state_.value, n)->getData();
      lstmValue.stateActiveValue =
          batchGrad_->getBatchValue(*preOutput_.value, n)->getData();
      lstmGrad.stateGrad =
          batchGrad_->getBatchValue(*state_.grad, n)->getData();
      lstmGrad.gateGrad = gateGrad->getData();
      lstmGrad.outputGrad = outputGrad->getData();
      {
        batchSize = outputGrad->getHeight();
        if (n != 0) {
          lstmValue.prevStateValue =
              batchGrad_->getBatchValue(*state_.value, n - 1)->getData();
          lstmGrad.prevStateGrad =
              batchGrad_->getBatchValue(*state_.grad, n - 1)->getData();
        } else {
          if (prevState_) {
            lstmValue.prevStateValue = totalState_->getData();
            lstmGrad.prevStateGrad = nullptr;
          } else {
            lstmValue.prevStateValue = nullptr;
            lstmGrad.prevStateGrad = nullptr;
          }
        }
        if (useGpu_) {
          LstmCompute::backwardBatch<1>(
              lstmValue, lstmGrad, getSize(), batchSize);
        } else {
          LstmCompute::backwardBatch<0>(
              lstmValue, lstmGrad, getSize(), batchSize);
        }
      }

      if (n != 0) {
        MatrixPtr tmp = batchGrad_->getBatchValue(n - 1, batchSize);
        tmp->mul(*gateGrad, *weightT, 1, 1);
      }

      if (n != 0 && weight_->getWGrad()) {
        /* backward weight */
        MatrixPtr outputValue = batchValue_->getBatchValue(n - 1, batchSize);
        weight_->getWGrad()->mul(*outputValue->getTranspose(), *gateGrad, 1, 1);
      } else if (prevOutput_ && weight_->getWGrad()) {
        weight_->getWGrad()->mul(
            *prevBatchOutput2_->getTranspose(), *gateGrad, 1, 1);
      }
    }
  }

  if (inputGrad) {
    batchGrad_->add(*inputGrad, *gate_.grad, /* seq2batch */ false);
  }
  if (bias_ && bias_->getWGrad()) {
    localBiasGrad_->collectBias(*gate_.grad, /* scale */ 1);
  }
}

void LstmLayer::forwardSeqParallel(int batchSize,
                                   size_t numSequences,
                                   const int *starts,
                                   MatrixPtr inputValue) {
  REGISTER_TIMER_INFO("LstmFwSeqParallelTime", getName().c_str());
  gate_.value->assign(*inputValue);
  if (bias_) {
    gate_.value->addBias(*localBias_, /* scale */ 1);
  }

  real *gateValue = gate_.value->getData();
  real *stateValue = state_.value->getData();
  real *outputValue = output_.value->getData();
  real *preOutputValue = preOutput_.value->getData();
  real *checkIg = checkIg_->getData();
  real *checkFg = checkFg_->getData();
  real *checkOg = checkOg_->getData();
  real *weight = weight_->getW()->getData();
  hl_lstm_parallel_forward(gateValue,
                           stateValue,
                           preOutputValue,
                           outputValue,
                           checkIg,
                           checkFg,
                           checkOg,
                           weight,
                           starts,
                           getSize(),
                           numSequences,
                           reversed_,
                           activeNode_,
                           activeGate_,
                           activeState_);
}

void LstmLayer::backwardSeqParallel(int batchSize,
                                    size_t numSequences,
                                    const int *starts,
                                    MatrixPtr inputGrad) {
  REGISTER_TIMER_INFO("LstmBwSeqParallelTime", getName().c_str());
  real *gateValue = gate_.value->getData();
  real *gateGrad = gate_.grad->getData();
  real *stateValue = state_.value->getData();
  real *stateGrad = state_.grad->getData();
  real *preOutputValue = preOutput_.value->getData();
  real *preOutputGrad = preOutput_.grad->getData();
  real *checkIg = checkIg_->getData();
  real *checkFg = checkFg_->getData();
  real *checkOg = checkOg_->getData();
  real *outputGrad = output_.grad->getData();
  real *weight = weight_->getW()->getData();

  real *checkIgGrad;
  real *checkFgGrad;
  real *checkOgGrad;
  if (bias_->getWGrad()) {
    checkIgGrad = checkIgGrad_->getData();
    checkFgGrad = checkFgGrad_->getData();
    checkOgGrad = checkOgGrad_->getData();
  } else {
    checkIgGrad = nullptr;
    checkFgGrad = nullptr;
    checkOgGrad = nullptr;
  }

  hl_lstm_parallel_backward_data(gateValue,
                                 gateGrad,
                                 stateValue,
                                 stateGrad,
                                 preOutputValue,
                                 preOutputGrad,
                                 outputGrad,
                                 checkIg,
                                 checkIgGrad,
                                 checkFg,
                                 checkFgGrad,
                                 checkOg,
                                 checkOgGrad,
                                 weight,
                                 starts,
                                 getSize(),
                                 numSequences,
                                 reversed_,
                                 activeNode_,
                                 activeGate_,
                                 activeState_);

  if (inputGrad) {
    inputGrad->add(*gate_.grad);
  }
  if (bias_ && bias_->getWGrad()) {
    localBiasGrad_->collectBias(*gate_.grad, 1);
  }

  real *outputValue = output_.value->getData();
  if (weight_->getWGrad()) {
    real *weightGrad = weight_->getWGrad()->getData();
    hl_lstm_parallel_backward_weight(weightGrad,
                                     outputValue,
                                     gateGrad,
                                     starts,
                                     getSize(),
                                     batchSize,
                                     numSequences,
                                     reversed_);
  }
}

}  // namespace paddle
