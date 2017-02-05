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

#include <gflags/gflags.h>
#include "Layer.h"
#include "SequenceToBatch.h"
#include "paddle/utils/Stat.h"

DEFINE_bool(rnn_use_batch, false, "Using the batch method for calculation.");

namespace paddle {

/**
 * @brief RecurrentLayer takes 1 input layer. The output size is the same with
 * input layer.
 * For each sequence [start, end] it performs the following computation:
 * \f[
 *    out_{i} = act(in_{i})     \      \      \text{for} \ i = start \\
 *    out_{i} = act(in_{i} + out_{i-1} * W) \ \ \text{for} \ start < i <= end
 *
 * \f]
 * If reversed is true, the order is reversed:
 * \f[
 *   out_{i} = act(in_{i})           \    \   \text{for} \ i = end  \\
 *   out_{i} = act(in_{i} + out_{i+1} * W) \ \ \text{for} \ start <= i < end
 * \f]
 * There are two methods to calculate rnn. One way is to compute rnn one
 * sequence by one sequence. The other way is to reorganize the input
 * into batches, then compute rnn one batch by one batch. Users can select
 * them by rnn_use_batch flag.
 */

class RecurrentLayer : public Layer {
public:
  explicit RecurrentLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;

  void backward(const UpdateCallback& callback) override;

  void resetState() override;

  void setState(LayerStatePtr state) override;

  LayerStatePtr getState() override;

protected:
  /**
   * @brief If user do not set --rnn_use_batch=true, it will
   * compute rnn forward one sequence by one sequence in default.
   * @param batchSize Total words number of all samples in this batch.
   * @param numSequences The sample number.
   * @param starts Each start position of each samples.
   */
  void forwardSequence(int batchSize, size_t numSequences, const int* starts);
  /**
   * @brief Compute rnn forward by one sequence.
   * @param start The start position of this sequence (or sample).
   * @param length The length of this sequence (or sample), namely the words
   * number of this sequence.
   */
  void forwardOneSequence(int start, int length);
  /**
   * @brief Compute rnn backward one sequence by onesequence.
   * @param batchSize Total words number of all samples in this batch.
   * @param numSequences The sample number.
   * @param starts Each start position of each samples.
   */
  void backwardSequence(int batchSize, size_t numSequences, const int* starts);
  /**
   * @brief Compute rnn backward by one sequence.
   * @param start The start position of this sequence (or sample).
   * @param length The length of this sequence (or sample), namely the words
   * number of this sequence.
   */
  void backwardOneSequence(int start, int length);

  /**
   * @brief Reorganize input into batches and compute rnn forward batch
   * by batch. It will convert batch shape to sequence after finishing forward.
   * The batch info can refer to SequenceToBatch class.
   * @param batchSize Total words number of all samples in this batch.
   * @param numSequences The sample number.
   * @param starts Each start position of each samples.
   */
  void forwardBatch(int batchSize, size_t numSequences, const int* starts);

  /**
   * @brief Reorganize input into batches and compute rnn forward batch
   * by batch.
   * @param batchSize Total words number of all samples in this batch.
   * @param numSequences The sample number.
   * @param starts Each start position of each samples.
   */
  void backwardBatch(int batchSize, size_t numSequences, const int* starts);

protected:
  std::unique_ptr<Weight> weight_;
  std::unique_ptr<Weight> bias_;

  /// frameOutput_[i] is used to hold the i-th sample of output_
  std::vector<Argument> frameOutput_;
  MatrixPtr prevOutput_;
  /// Whether compute rnn by reverse.
  bool reversed_;
  /// If compute batch by batch, batchValue_ will be used to save the
  /// reorganized input value.
  std::unique_ptr<SequenceToBatch> batchValue_;
  /// If compute batch by batch, batchGrad_ will be used to save the
  /// gradient with respect to reorganized input value.
  std::unique_ptr<SequenceToBatch> batchGrad_;
};

REGISTER_LAYER(recurrent, RecurrentLayer);

bool RecurrentLayer::init(const LayerMap& layerMap,
                          const ParameterMap& parameterMap) {
  if (!Layer::init(layerMap, parameterMap)) return false;
  CHECK_EQ(1U, inputLayers_.size());
  CHECK_EQ(1U, parameters_.size());
  CHECK_EQ(getSize() * getSize(), parameters_[0]->getSize());
  weight_.reset(new Weight(getSize(), getSize(), parameters_[0]));
  if (biasParameter_.get() != NULL) {
    bias_.reset(new Weight(1, getSize(), biasParameter_));
  }
  reversed_ = config_.reversed();
  return true;
}

void RecurrentLayer::resetState() {
  CHECK(!reversed_) << "state is not allowed for reversed recurrent layer";
  Matrix::resizeOrCreate(
      prevOutput_, 1, getSize(), /* trans= */ false, useGpu_);
  prevOutput_->zeroMem();
}

void RecurrentLayer::setState(LayerStatePtr state) {
  CHECK(state->value.size() == 1) << "one matrix is expected for RNN state";
  prevOutput_->copyFrom(*(state->value[0]));
}

LayerStatePtr RecurrentLayer::getState() {
  LayerStatePtr res = std::make_shared<LayerState>();
  res->value.push_back(prevOutput_->clone(0, 0, useGpu_));
  res->value[0]->copyFrom(*prevOutput_);
  return res;
}

void RecurrentLayer::forward(PassType passType) {
  REGISTER_TIMER_INFO("RecurrentFwTimer", getName().c_str());
  Layer::forward(passType);
  const Argument& input = getInput(0);
  CHECK(input.sequenceStartPositions);
  int batchSize = input.getBatchSize();
  size_t numSequences = input.getNumSequences();
  resetOutput(batchSize, getSize());
  CHECK_EQ(getSize(), input.value->getWidth());
  const int* starts = input.sequenceStartPositions->getData(false);
  CHECK_EQ(starts[numSequences], batchSize);

  output_.value->assign(*input.value);
  if (bias_) {
    output_.value->addBias(*bias_->getW(), 1);
  }
  if (!FLAGS_rnn_use_batch) {
    forwardSequence(batchSize, numSequences, starts);
  } else {
    forwardBatch(batchSize, numSequences, starts);
  }
}

void RecurrentLayer::forwardSequence(int batchSize,
                                     size_t numSequences,
                                     const int* starts) {
  REGISTER_TIMER_INFO("RecurrentFwSequence", getName().c_str());
  frameOutput_.reserve(batchSize);
  for (int i = frameOutput_.size(); i < batchSize; ++i) {
    Argument arg;
    arg.value = Matrix::create(nullptr,
                               /* height= */ 1,
                               getSize(),
                               /* trans= */ false,
                               useGpu_);
    arg.grad = Matrix::create(nullptr,
                              /* height= */ 1,
                              getSize(),
                              /* trans= */ false,
                              useGpu_);
    frameOutput_.push_back(arg);
  }

  for (int i = 0; i < batchSize; ++i) {
    frameOutput_[i].value->setData(output_.value->getData() + i * getSize());
  }

  AsyncGpuBlock asyncGpuBlock;
  for (size_t i = 0; i < numSequences; ++i) {
    forwardOneSequence(starts[i], starts[i + 1] - starts[i]);
  }
}

void RecurrentLayer::forwardOneSequence(int start, int length) {
  if (!reversed_) {
    if (prevOutput_) {
      frameOutput_[start].value->mul(*prevOutput_, *weight_->getW(), 1, 1);
    }
    activation_->forward(frameOutput_[start]).check();

    for (int i = 1; i < length; ++i) {
      frameOutput_[start + i].value->mul(
          *frameOutput_[start + i - 1].value, *weight_->getW(), 1, 1);
      activation_->forward(frameOutput_[start + i]).check();
    }
    if (prevOutput_) {
      prevOutput_->assign(*frameOutput_[start + length - 1].value);
    }
  } else {
    activation_->forward(frameOutput_[start + length - 1]).check();
    for (int i = length - 2; i >= 0; --i) {
      frameOutput_[start + i].value->mul(
          *frameOutput_[start + i + 1].value, *weight_->getW(), 1, 1);
      activation_->forward(frameOutput_[start + i]).check();
    }
  }
}

void RecurrentLayer::backward(const UpdateCallback& callback) {
  REGISTER_TIMER_INFO("RecurrentBwTimer", getName().c_str());
  const Argument& input = getInput(0);
  CHECK(input.sequenceStartPositions);
  int batchSize = input.getBatchSize();
  const int* starts = input.sequenceStartPositions->getData(false);
  size_t numSequences = input.getNumSequences();

  if (!FLAGS_rnn_use_batch) {
    backwardSequence(batchSize, numSequences, starts);
  } else {
    backwardBatch(batchSize, numSequences, starts);
  }

  if (input.grad) {
    input.grad->add(*output_.grad);
  }

  if (bias_ && bias_->getWGrad()) {
    bias_->getWGrad()->collectBias(*output_.grad, 1);
    bias_->getParameterPtr()->incUpdate(callback);
  }

  weight_->getParameterPtr()->incUpdate(callback);
}

void RecurrentLayer::backwardSequence(int batchSize,
                                      size_t numSequences,
                                      const int* starts) {
  REGISTER_TIMER_INFO("RecurrentBwSequence", getName().c_str());
  for (int i = 0; i < batchSize; ++i) {
    frameOutput_[i].grad->setData(output_.grad->getData() + i * getSize());
  }

  AsyncGpuBlock asyncGpuBlock;
  for (size_t i = 0; i < numSequences; ++i) {
    backwardOneSequence(starts[i], starts[i + 1] - starts[i]);
  }
}

void RecurrentLayer::backwardOneSequence(int start, int length) {
  MatrixPtr weightT = weight_->getW()->getTranspose();
  if (!reversed_) {
    for (int i = length - 1; i > 0; --i) {
      activation_->backward(frameOutput_[start + i]).check();
      frameOutput_[start + i - 1].grad->mul(
          *frameOutput_[start + i].grad, *weightT, 1, 1);
    }
    activation_->backward(frameOutput_[start]).check();
    if (weight_->getWGrad()) {
      weight_->getWGrad()->mul(
          *output_.value->subMatrix(start, length - 1)->getTranspose(),
          *output_.grad->subMatrix(start + 1, length - 1),
          1,
          1);
    }
  } else {
    for (int i = 0; i < length - 1; ++i) {
      activation_->backward(frameOutput_[start + i]).check();
      frameOutput_[start + i + 1].grad->mul(
          *frameOutput_[start + i].grad, *weightT, 1, 1);
    }
    activation_->backward(frameOutput_[start + length - 1]).check();
    if (weight_->getWGrad()) {
      weight_->getWGrad()->mul(
          *output_.value->subMatrix(start + 1, length - 1)->getTranspose(),
          *output_.grad->subMatrix(start, length - 1),
          1,
          1);
    }
  }
}

void RecurrentLayer::forwardBatch(int batchSize,
                                  size_t numSequences,
                                  const int* starts) {
  if (!batchValue_) {
    batchValue_.reset(new SequenceToBatch(useGpu_));
  }

  batchValue_->resizeOrCreateBatch(batchSize, numSequences, starts, reversed_);

  batchValue_->copyFromSeq(*output_.value);
  {
    REGISTER_TIMER_INFO("RecurrentFwBatch", getName().c_str());
    AsyncGpuBlock asyncGpuBlock;
    /* forward one batch */
    for (size_t n = 0; n < batchValue_->getNumBatch(); n++) {
      MatrixPtr batch2 = batchValue_->getBatchValue(n);

      if (n != 0) {
        MatrixPtr batch1 =
            batchValue_->getBatchValue(n - 1, batch2->getHeight());
        batch2->mul(*batch1, *weight_->getW(), 1, 1);
      }
      Argument arg;
      arg.value = batch2;
      activation_->forward(arg).check();
    }
  }
  batchValue_->copyBackSeq(*output_.value);
}

void RecurrentLayer::backwardBatch(int batchSize,
                                   size_t numSequences,
                                   const int* starts) {
  if (!batchGrad_) {
    batchGrad_.reset(new SequenceToBatch(useGpu_));
  }
  batchGrad_->shareIndexWith(*batchValue_);

  size_t numBatch = batchGrad_->getNumBatch();
  bool backwardByBatch = numBatch < numSequences;

  batchGrad_->copyFromSeq(*output_.grad);
  {
    REGISTER_TIMER_INFO("RecurrentBwData", getName().c_str());
    MatrixPtr weightT = weight_->getW()->getTranspose();
    AsyncGpuBlock asyncGpuBlock;
    /* backward one batch */
    for (int n = (int)numBatch - 1; n >= 0; n--) {
      MatrixPtr batch2 = batchGrad_->getBatchValue(n);
      MatrixPtr batch1 = batchValue_->getBatchValue(n, batch2->getHeight());

      Argument arg;
      arg.value = batch1;
      arg.grad = batch2;
      activation_->backward(arg).check();

      if (n != 0) {
        batch1 = batchGrad_->getBatchValue(n - 1, batch2->getHeight());
        batch1->mul(*batch2, *weightT, 1, 1);
      }

      if (backwardByBatch && weight_->getWGrad()) {
        if (n != 0) {
          /* backward weight */
          batch1 = batchValue_->getBatchValue(n - 1, batch2->getHeight());
          weight_->getWGrad()->mul(*batch1->getTranspose(), *batch2, 1, 1);
        }
      }
    }
  }

  batchGrad_->copyBackSeq(*output_.grad);

  if (!backwardByBatch && weight_->getWGrad()) {
    REGISTER_TIMER_INFO("RecurrentBwWeight", getName().c_str());
    AsyncGpuBlock asyncGpuBlock;
    for (size_t seq = 0; seq < numSequences; ++seq) {
      int len = starts[seq + 1] - starts[seq];
      if (!reversed_) {
        weight_->getWGrad()->mul(
            *output_.value->subMatrix(starts[seq], len - 1)->getTranspose(),
            *output_.grad->subMatrix(starts[seq] + 1, len - 1),
            1,
            1);
      } else {
        weight_->getWGrad()->mul(
            *output_.value->subMatrix(starts[seq] + 1, len - 1)->getTranspose(),
            *output_.grad->subMatrix(starts[seq], len - 1),
            1,
            1);
      }
    }
  }
}

}  // namespace paddle
