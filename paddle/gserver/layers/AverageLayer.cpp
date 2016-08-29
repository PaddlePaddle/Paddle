/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "AverageLayer.h"

#include "paddle/utils/Logging.h"

#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(average, AverageLayer);

bool AverageLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }
  dataMtx_ = Matrix::create(nullptr, 1, 1, false, useGpu_);
  outMtx_ = Matrix::create(nullptr, 1, getSize(), false, useGpu_);
  // average strategy
  if (config_.average_strategy() == "average") {
    mode_ = kAverage;
  } else if (config_.average_strategy() == "sum") {
    mode_ = kSum;
  } else if (config_.average_strategy() == "squarerootn") {
    mode_ = kAverageSquareRootN;
  } else {
    LOG(FATAL) << "Unknown average strategy: " << config_.average_strategy();
  }
  // transform to which sequence type
  if (config_.trans_type() == "non-seq") {
    type_ = kNonSeq;
  } else if (config_.trans_type() == "seq") {
    type_ = kSeq;
  } else {
    LOG(FATAL) << "Unknown trans_type: " << config_.trans_type();
  }
  setNeedSequenceInfo(false);
  return true;
}

void AverageLayer::forward(PassType passType) {
  Layer::forward(passType);

  // average layer should have exactly 1 input
  CHECK_EQ(1U, inputLayers_.size());

  size_t dim = getSize();
  const Argument& input = getInput(0);
  int64_t newBatchSize =
      type_ ? input.getNumSubSequences() : input.getNumSequences();
  ICpuGpuVectorPtr startPositions =
      type_ ? input.subSequenceStartPositions
            : input.sequenceStartPositions;
  const int* starts = startPositions->getData(false);
  size_t numSequences = startPositions->getSize() - 1;

  // check
  CHECK_EQ(numSequences, (size_t)newBatchSize);
  CHECK_EQ(starts[numSequences], input.getBatchSize());
  if (type_) {
    // when trans_type = seq, input must hasSubseq
    CHECK_EQ(input.hasSubseq(), 1UL);
  }

  CHECK_EQ(dim, input.value->getWidth());

  resetOutput(newBatchSize, dim);
  auto startsPos = startPositions->getVector(useGpu_);
  MatrixPtr inputValue = getInputValue(0);
  getOutputValue()->sequenceAvgForward(*inputValue, *startsPos, mode_);

  /* If type_ = kNonSeq, both seq has or not has sub-seq degrade to a non-seq,
   * thus, in this case, output_ has no sequenceStartPositions.
   * If type_ = kSeq, seq has sub-seq degrades to a seq, thus, only in this
   * case, we should compute the new sequenceStartPositions.
  */
  if (type_) {
    output_.degradeSequence(input, useGpu_);
  }

  /* add the bias-vector AFTER average operation */
  if (biases_.get() != NULL) {
    MatrixPtr outV = getOutputValue();
    outV->addBias(*(biases_->getW()), 1);
  }

  /* activation */ { forwardActivation(); }
}

void AverageLayer::backward(const UpdateCallback& callback) {
  const Argument& input = getInput(0);
  ICpuGpuVectorPtr startPositions =
      type_ ? input.subSequenceStartPositions
            : input.sequenceStartPositions;
  const int* starts = startPositions->getData(false);
  /* Do derivation */ { backwardActivation(); }

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    // Increasing the number of gradient
    biases_->getParameterPtr()->incUpdate(callback);
  }

  MatrixPtr grad = getInputGrad(0);
  if (grad) {
    size_t dim = getSize();
    real* gradientData = getInputGrad(0)->getData();
    real* gradient = getOutputGrad()->getData();
    size_t numSequences = startPositions->getSize() - 1;
    for (size_t sequenceId = 0; sequenceId < numSequences; ++sequenceId) {
      // TODO(Dangqingqing) optimization for GPU
      int sequenceLength = starts[sequenceId + 1] - starts[sequenceId];
      if (0 == sequenceLength) {
        // empty sequence
        continue;
      }
      dataMtx_->setData(gradientData + starts[sequenceId] * dim, sequenceLength,
                        dim);
      outMtx_->setData(gradient + sequenceId * dim);
      switch (mode_) {
        case kAverage: {
          // plain average
          dataMtx_->addBias(*outMtx_, 1.0f / sequenceLength);
          break;
        }
        case kSum: {
          // sum instead of average
          dataMtx_->addBias(*outMtx_, 1.0f);
          break;
        }
        case kAverageSquareRootN: {
          // divide by square root of sequenceLength
          dataMtx_->addBias(*outMtx_, 1.0f / sqrt(sequenceLength));
          break;
        }
        default: { LOG(FATAL) << "should not reach here"; }
      }
    }
  }
}

}  // namespace paddle
