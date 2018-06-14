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

#include "ExpandLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(expand, ExpandLayer);

bool ExpandLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(inputLayers_.size(), 2UL);
  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }
  // which sequence type of input[0]
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

void ExpandLayer::forward(PassType passType) {
  Layer::forward(passType);
  // Expand layer should have exactly 2 input, one for data, one for size
  CHECK_EQ(2U, inputLayers_.size());

  // using two input:
  // * first one for data;
  // * second one only for sequence info
  const Argument& shapeInput = getInput(1);
  const Argument& dataInput = getInput(0);
  size_t outputBatchSize = shapeInput.getBatchSize();
  auto startPositions = type_ ? shapeInput.subSequenceStartPositions
                              : shapeInput.sequenceStartPositions;
  size_t numSequences = startPositions->getSize() - 1;
  const int* starts = startPositions->getData(false);

  CHECK_EQ(starts[numSequences], shapeInput.getBatchSize());
  if (type_) {
    // when trans_type = seq, input[1] must hasSubseq
    CHECK_EQ(shapeInput.hasSubseq(), 1UL);
    CHECK_EQ(dataInput.getNumSequences(), shapeInput.getNumSequences());
  } else {
    CHECK_EQ(dataInput.getBatchSize(), shapeInput.getNumSequences());
  }

  // set output sequence info as shape sequence
  output_.sequenceStartPositions = shapeInput.sequenceStartPositions;
  if (shapeInput.hasSubseq()) {
    output_.subSequenceStartPositions = shapeInput.subSequenceStartPositions;
  }

  // reserve output: Expand output to batchsize of sequence data.
  reserveOutput(outputBatchSize, dataInput.value->getWidth());

  MatrixPtr inputValue = getInputValue(0);
  MatrixPtr outputValue = getOutputValue();

  ICpuGpuVector::resizeOrCreate(expandStartsPos_, outputBatchSize, false);
  int* expandStarts = expandStartsPos_->getMutableData(false);
  for (size_t sequenceId = 0; sequenceId < numSequences; ++sequenceId) {
    int sequenceLength = starts[sequenceId + 1] - starts[sequenceId];
    for (int j = 0; j < sequenceLength; j++) {
      expandStarts[starts[sequenceId] + j] = sequenceId;
    }
  }

  outputValue->copyByRowIndex(*inputValue,
                              *expandStartsPos_->getVector(useGpu_));

  if (biases_.get() != NULL) {
    outputValue->addBias(*(biases_->getW()), 1);
  }
}

void ExpandLayer::backward(const UpdateCallback& callback) {
  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);
    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  if (!getInputGrad(0)) return;
  MatrixPtr inputGrad = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();
  auto cpuSeqStartPos = type_ ? getInput(1).subSequenceStartPositions
                              : getInput(1).sequenceStartPositions;
  size_t numSequences = cpuSeqStartPos->getSize() - 1;
  const int* starts = cpuSeqStartPos->getData(false);

  CHECK_EQ(inputGrad->getWidth(), outputGrad->getWidth());
  CHECK_EQ(outputGrad->getHeight(), (size_t)starts[numSequences]);

  AsyncGpuBlock asyncGpuBlock;

  // sum to get the grad
  real scale = 1;
  for (size_t sequenceId = 0; sequenceId < numSequences; sequenceId++) {
    // TODO(Dangqingqing) optimization for GPU
    int sequenceLength = starts[sequenceId + 1] - starts[sequenceId];
    if (sequenceLength == 0) {
      // empty sequence
      continue;
    }
    MatrixPtr copyData = inputGrad->subMatrix(sequenceId, 1);
    copyData->collectBias(
        *outputGrad->subMatrix(starts[sequenceId], sequenceLength), scale);
  }
}

}  // namespace paddle
