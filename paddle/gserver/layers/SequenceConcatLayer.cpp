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

#include "Layer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * A layer for concatenating the first sequence with the second sequence
 * Input: two sequences each containing the same number of instances
 *        seq1 = [a1, a2, ..., an]
 *        seq2 = [b1, b2, ..., bn]
 * Output: a concatenated sequence of the two input sequences
 *        out = [a1, b1, a2, b2, ..., an, bn]
 */

class SequenceConcatLayer : public Layer {
 protected:
  std::unique_ptr<Weight> biases_;

 public:
  explicit SequenceConcatLayer(const LayerConfig& config) : Layer(config) {}

  ~SequenceConcatLayer() {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(seqconcat, SequenceConcatLayer);

bool SequenceConcatLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  // sequene concatenation layer should have exactly 2 inputs
  CHECK_EQ(2U, inputLayers_.size());

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }

  setNeedSequenceInfo(false);
  return true;
}

void SequenceConcatLayer::forward(PassType passType) {
  Layer::forward(passType);

  size_t dim = getSize();

  const Argument& input1 = getInput(0);
  size_t numSequences1 = input1.getNumSequences();
  auto startPositions1 = input1.sequenceStartPositions->getVector(false);

  const Argument& input2 = getInput(1);
  size_t numSequences2 = input2.getNumSequences();
  auto startPositions2 = input2.sequenceStartPositions->getVector(false);

  CHECK_EQ(dim, input1.value->getWidth());
  CHECK_EQ(startPositions1->getData()[numSequences1], input1.getBatchSize());
  CHECK_EQ(numSequences1, startPositions1->getSize() - 1);

  CHECK_EQ(dim, input2.value->getWidth());
  CHECK_EQ(startPositions2->getData()[numSequences2], input2.getBatchSize());
  CHECK_EQ(numSequences2, startPositions2->getSize() - 1);

  CHECK_EQ(numSequences1, numSequences2);

  MatrixPtr inputValue1 = getInputValue(0);
  MatrixPtr inputValue2 = getInputValue(1);

  // reset output
  reserveOutput(inputValue1->getHeight() + inputValue2->getHeight(), dim);

  MatrixPtr outputValue = getOutputValue();

  const int* starts1 = startPositions1->getData();
  const int* starts2 = startPositions2->getData();

  {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SequenceConcatLayerForward", getName().c_str());

    size_t offset = 0;
    size_t leftNumIns = 0;
    size_t rightNumIns = 0;
    for (size_t seqId = 0; seqId < numSequences1; ++seqId) {
      leftNumIns = starts1[seqId + 1] - starts1[seqId];
      outputValue->subMatrix(offset, leftNumIns)
          ->assign(*(inputValue1->subMatrix(starts1[seqId], leftNumIns)));
      offset += leftNumIns;

      rightNumIns = starts2[seqId + 1] - starts2[seqId];
      outputValue->subMatrix(offset, rightNumIns)
          ->assign(*(inputValue2->subMatrix(starts2[seqId], rightNumIns)));
      offset += rightNumIns;
    }

    // modify the sequenceStartPositions
    ICpuGpuVector::resizeOrCreate(
        output_.sequenceStartPositions, numSequences1 + 1, false);

    int* tgtBuf = output_.sequenceStartPositions->getMutableData(false);

    for (size_t seqId = 0; seqId < numSequences1 + 1; ++seqId) {
      tgtBuf[seqId] = starts1[seqId] + starts2[seqId];
    }
  }

  if (biases_.get() != NULL) {
    MatrixPtr outV = getOutputValue();
    outV->addBias(*(biases_->getW()), 1);
  }

  /* activation */
  forwardActivation();
}

void SequenceConcatLayer::backward(const UpdateCallback& callback) {
  /* activation */
  backwardActivation();

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    // Increasing the number of gradient
    biases_->getParameterPtr()->incUpdate(callback);
  }

  MatrixPtr inputGrad1 = getInputGrad(0);
  MatrixPtr inputGrad2 = getInputGrad(1);
  MatrixPtr outputGrad = getOutputGrad();
  auto startPositions1 = getInput(0).sequenceStartPositions->getVector(false);
  auto startPositions2 = getInput(1).sequenceStartPositions->getVector(false);

  size_t numSequences1 = startPositions1->getSize() - 1;
  size_t numSequences2 = startPositions2->getSize() - 1;

  CHECK_EQ(numSequences1, numSequences2);

  const int* starts1 = startPositions1->getData();
  const int* starts2 = startPositions2->getData();

  {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SequenceConcatLayerBackward", getName().c_str());

    size_t offset = 0;
    size_t leftNumIns = 0;
    size_t rightNumIns = 0;
    for (size_t seqId = 0; seqId < numSequences1; ++seqId) {
      leftNumIns = starts1[seqId + 1] - starts1[seqId];
      if (inputGrad1) {
        inputGrad1->subMatrix(starts1[seqId], leftNumIns)
            ->add(*(outputGrad->subMatrix(offset, leftNumIns)));
      }
      offset += leftNumIns;

      rightNumIns = starts2[seqId + 1] - starts2[seqId];
      if (inputGrad2) {
        inputGrad2->subMatrix(starts2[seqId], rightNumIns)
            ->add(*(outputGrad->subMatrix(offset, rightNumIns)));
      }
      offset += rightNumIns;
    }
  }
}

}  // namespace paddle
