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

#include "MKLPackedLstmLayer.h"

DECLARE_bool(prev_batch_state);

namespace paddle {

REGISTER_LAYER(mkl_packed_lstmemory, MKLPackedLstmLayer);

bool MKLPackedLstmLayer::init(const LayerMap &layerMap,
                              const ParameterMap &parameterMap) {
  if (!LstmLayer::init(layerMap, parameterMap)) return false;
  packedWeight_.reset(new MKLPackedWeight(weight_->getW()));
  packedWeight_->pack();
  return true;
}

void MKLPackedLstmLayer::forward(PassType passType) {
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

  Matrix::resizeOrCreate(gate_.value, batchSize, getSize() * 4, false, false);
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
                           false,
                           false);
    state_.value = Matrix::create(nullptr, batchSize, getSize(), false, false);
    state_.value->setData(totalState_->getData() +
                          prevState_->getHeight() * getSize());
  } else {
    Matrix::resizeOrCreate(state_.value, batchSize, getSize(), false, false);
  }
  Matrix::resizeOrCreate(preOutput_.value, batchSize, getSize(), false, false);

  if (!useBatch_) {
    LstmLayer::forwardSequence(batchSize, numSequences, starts, input.value);
  } else {
    forwardBatch(batchSize, numSequences, starts, input.value);
  }
  { forwardActivation(); }
}

void MKLPackedLstmLayer::forwardBatch(int batchSize,
                                      size_t numSequences,
                                      const int *starts,
                                      MatrixPtr inputValue) {
  REGISTER_TIMER_INFO("LstmFwBatchTime", getName().c_str());

  hl_lstm_value lstmValue;
  lstmValue.checkIg = checkIg_->getData();
  lstmValue.checkFg = checkFg_->getData();
  lstmValue.checkOg = checkOg_->getData();

  if (!batchValue_) {
    batchValue_.reset(new SequenceToBatch(false));
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
        MatrixPtr preBatchValue = batchValue_->getBatchValue(n - 1, batchSize);
        packedWeight_->gemm_compute(preBatchValue, gateValue);
      } else if (prevOutput_) {
        Matrix::resizeOrCreate(
            prevBatchOutput2_, gateValue->getHeight(), getSize(), false, false);
        batchValue_->prevOutput2Batch(*prevOutput_, *prevBatchOutput2_);
        packedWeight_->gemm_compute(prevBatchOutput2_, gateValue);

        batchValue_->prevOutput2Batch(*prevState_,
                                      *totalState_->subMatrix(0, numSequences));
      }

      lstmValue.gateValue = gateValue->getData();
      lstmValue.outputValue = outputValue->getData();
      lstmValue.stateValue =
          batchValue_->getBatchValue(*state_.value, n)->getData();
      lstmValue.stateActiveValue =
          batchValue_->getBatchValue(*preOutput_.value, n)->getData();
      { LstmCompute::forwardBatch<0>(lstmValue, getSize(), batchSize); }
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

void MKLPackedLstmLayer::backward(const UpdateCallback &callback) {
  REGISTER_TIMER_INFO("MKLPackedLstmBwTimer", getName().c_str());
  LstmLayer::backward(callback);
  packedWeight_->pack();
}

}  // namespace paddle
