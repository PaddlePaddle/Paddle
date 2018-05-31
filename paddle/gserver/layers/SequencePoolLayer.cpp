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

#include "SequencePoolLayer.h"
#include "paddle/utils/Logging.h"

namespace paddle {

bool SequencePoolLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  // seqlastins/max/average layer should have exactly 1 input
  CHECK_EQ(1U, inputLayers_.size());

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
  stride_ = config_.seq_pool_stride();
  setNeedSequenceInfo(false);
  return true;
}

void SequencePoolLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& input = getInput(0);
  CHECK(input.hasSeq() || input.hasSubseq())
      << "Input should be a sequence or subsequence for layer " << getName();

  newBatchSize_ = type_ ? input.getNumSubSequences() : input.getNumSequences();
  size_t dim = getSize();
  // check
  CHECK_EQ(dim, input.value->getWidth());
  startPositions_ =
      type_ ? input.subSequenceStartPositions : input.sequenceStartPositions;
  auto starts = startPositions_->getVector(false);
  CHECK_EQ(starts->getData()[newBatchSize_], input.getBatchSize());
  CHECK_EQ(newBatchSize_, starts->getSize() - 1);

  /* If type_ = kNonSeq, both seq has or not has sub-seq degrade to a non-seq,
   * thus, in this case, output_ has no sequenceStartPositions.
   * If type_ = kSeq, seq has sub-seq degrades to a seq, thus, only in this
   * case, we should compute the new sequenceStartPositions.
   */
  if (type_) {
    CHECK(input.subSequenceStartPositions)
        << "when trans_type = seq, input must hasSubseq";
    output_.degradeSequence(input);
  }
  if (stride_ > 0) {
    CHECK_EQ(input.hasSubseq(), 0UL)
        << "sequence stride pooling is invalid for hasSubseq now";
    output_.poolSequenceWithStride(input, stride_, &startPositions_, reversed_);
    newBatchSize_ = startPositions_->getSize() - 1;
  }

  resetOutput(newBatchSize_, dim);
}

void SequencePoolLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ { backwardActivation(); }

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    // Increasing the number of gradient
    biases_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
