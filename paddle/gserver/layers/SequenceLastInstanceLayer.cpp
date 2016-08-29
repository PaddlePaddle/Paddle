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


#include "paddle/utils/Logging.h"

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * A layer for extracting the last instance of the input sequence.
 * Input: a sequence
 * If SequenceLevel = kNonseq:
 *   Output: a sequence containing only the last instance of the input sequence
 * If SequenceLevel = kSeq:
 *   Check input sequence must has sub-sequence
 *   Output: a sequence containing only the last instance of each sub-sequence
 * of the input sequence
 */

class SequenceLastInstanceLayer : public Layer {
protected:
  std::unique_ptr<Weight> biases_;
  MatrixPtr tmpSrc_;
  MatrixPtr tmpDest_;
  enum SequenceLevel { kNonSeq = 0, kSeq = 1 };
  int type_;

public:
  explicit SequenceLastInstanceLayer(const LayerConfig& config)
      : Layer(config) {}

  ~SequenceLastInstanceLayer() {}

  bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  void forward(PassType passType);
  void backward(const UpdateCallback& callback = nullptr);
};

REGISTER_LAYER(seqlastins, SequenceLastInstanceLayer);

bool SequenceLastInstanceLayer::init(const LayerMap& layerMap,
                                     const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  // seqlastins layer should have exactly 1 input
  CHECK_EQ(1U, inputLayers_.size());

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }

  tmpSrc_ =
      Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, useGpu_);
  tmpDest_ =
      Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, useGpu_);

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

void SequenceLastInstanceLayer::forward(PassType passType) {
  Layer::forward(passType);

  size_t dim = getSize();
  const Argument& input = getInput(0);

  // check
  auto startPositions =
      type_ ? input.subSequenceStartPositions->getVector(false)
            : input.sequenceStartPositions->getVector(false);
  size_t height = type_ ? input.getNumSubSequences() : input.getNumSequences();
  CHECK_EQ(dim, input.value->getWidth());
  CHECK_EQ(startPositions->getData()[height], input.getBatchSize());
  CHECK_EQ(height, startPositions->getSize() - 1);
  if (type_) {
    // when trans_type = seq, input must hasSubseq
    CHECK_EQ(input.hasSubseq(), 1UL);
  }

  reserveOutput(height, dim);
  const int* starts = startPositions->getData();
  MatrixPtr inputValue = getInputValue(0);
  MatrixPtr outputValue = getOutputValue();

  {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SequenceLastInstanceLayerForward", getName().c_str());

    for (size_t seqId = 0; seqId < height; ++seqId) {
      int insId =
          config_.select_first() ? starts[seqId] : starts[seqId + 1] - 1;

      outputValue->subMatrix(seqId, 1, tmpDest_)
          ->assign(*(inputValue->subMatrix(insId, 1, tmpSrc_)));
    }
    /* If type_ = kNonSeq, both seq has or not has sub-seq degrade to a non-seq,
     * thus, in this case, output_ has no sequenceStartPositions.
     * If type_ = kSeq, seq has sub-seq degrades to a seq, thus, only in this
     * case, we should compute the new sequenceStartPositions.
    */
    if (type_) {
      output_.degradeSequence(input, useGpu_);
    }
  }

  if (biases_.get() != NULL) {
    outputValue->addBias(*(biases_->getW()), 1);
  }

  /*  activation, should set to 'linear' in most cases */
  forwardActivation();
}

void SequenceLastInstanceLayer::backward(const UpdateCallback& callback) {
  /* activation, should set to 'linear' in most cases */
  backwardActivation();

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    // Increasing the number of gradient
    biases_->getParameterPtr()->incUpdate(callback);
  }

  MatrixPtr inputGrad = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();
  auto startPositions =
      type_ ? getInput(0).subSequenceStartPositions->getVector(false)
            : getInput(0).sequenceStartPositions->getVector(false);
  const int* starts = startPositions->getData();
  size_t numSequences = startPositions->getSize() - 1;

  if (inputGrad) {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SequenceLastInstanceLayerBackward", getName().c_str());

    for (size_t seqId = 0; seqId < numSequences; ++seqId) {
      int insId =
          config_.select_first() ? starts[seqId] : starts[seqId + 1] - 1;

      inputGrad->subMatrix(insId, 1, tmpDest_)
          ->add(*(outputGrad->subMatrix(seqId, 1, tmpSrc_)));
    }
  }
}

}  // namespace paddle
