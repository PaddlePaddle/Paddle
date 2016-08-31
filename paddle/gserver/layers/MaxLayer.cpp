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


#include "MaxLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(max, MaxLayer);

bool MaxLayer::init(const LayerMap& layerMap,
                    const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
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

void MaxLayer::forward(PassType passType) {
  Layer::forward(passType);
  // max layer should have exactly 1 input
  CHECK_EQ(1U, inputLayers_.size());

  size_t dim = getSize();
  const Argument& input = getInput(0);
  int64_t newBatchSize =
      type_ ? input.getNumSubSequences() : input.getNumSequences();
  ICpuGpuVectorPtr startPositions =
      type_ ? input.subSequenceStartPositions
            : input.sequenceStartPositions;
  auto starts = startPositions->getVector(useGpu_);
  size_t numSequences = startPositions->getSize() - 1;

  CHECK_EQ(dim, input.value->getWidth());
  CHECK_EQ(numSequences, (size_t)newBatchSize);
  CHECK_EQ(startPositions->getData(false)[numSequences], input.getBatchSize());
  if (type_) {
    // when trans_type = seq, input must hasSubseq
    CHECK_EQ(input.hasSubseq(), 1UL);
  }

  // reset output: resize to "num of sequences", not "batch size".
  resetOutput(newBatchSize, dim);

  IVector::resizeOrCreate(maxIndex_, newBatchSize * dim, useGpu(deviceId_));
  maxIndex_->zeroMem();

  MatrixPtr inputValue = getInputValue(0);
  MatrixPtr outputValue = getOutputValue();

  {
    REGISTER_TIMER_INFO("MaxLayerForward", getName().c_str());
    outputValue->maxSequenceForward(*inputValue, *starts, *maxIndex_);
  }

  /* If type_ = kNonSeq, both seq has or not has sub-seq degrade to a non-seq,
   * thus, in this case, output_ has no cpuSequenceStartPositions.
   * If type_ = kSeq, seq has sub-seq degrades to a seq, thus, only in this
   * case, we should compute the new cpuSequenceStartPositions.
  */
  if (type_) {
    output_.degradeSequence(input, useGpu_);
  }

  if (config_.output_max_index()) {
    // copy maxIndex_ to output
    outputValue->copyFrom(*maxIndex_);
  } else {
    /* add the bias-vector AFTER max operation */
    if (biases_.get() != NULL) {
      outputValue->addBias(*(biases_->getW()), 1);
    }
    /* activation */ { forwardActivation(); }
  }
}

void MaxLayer::backward(const UpdateCallback& callback) {
  CHECK(!config_.output_max_index())
      << "backward is not available when output_max_index is set";
  /* Do derivation */ { backwardActivation(); }

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    // Increasing the number of gradient
    biases_->getParameterPtr()->incUpdate(callback);
  }

  MatrixPtr inputGrad = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();
  if (inputGrad) {
    ICpuGpuVectorPtr starts =
        type_ ? getInput(0).subSequenceStartPositions
              : getInput(0).sequenceStartPositions;
    REGISTER_TIMER_INFO("MaxLayerBackward", getName().c_str());
    inputGrad->maxSequenceBackward(*outputGrad,
        *(starts->getVector(useGpu_)), *maxIndex_);
  }
}

}  // namespace paddle
