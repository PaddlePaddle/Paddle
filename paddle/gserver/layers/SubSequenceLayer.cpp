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
#include "paddle/math/Vector.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * A layer for taking the subsequence according to given offset and size
 * Input: original sequence, offset, size
 * Output: subsequence
 */

class SubSequenceLayer : public Layer {
 protected:
  std::unique_ptr<Weight> biases_;
  MatrixPtr tmpSrc_;
  MatrixPtr tmpDest_;

 public:
  explicit SubSequenceLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(subseq, SubSequenceLayer);

bool SubSequenceLayer::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  // sequene concatenation layer should have exactly 2 inputs
  CHECK_EQ(3U, inputLayers_.size());

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }

  tmpSrc_ =
      Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, useGpu_);
  tmpDest_ =
      Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, useGpu_);

  setNeedSequenceInfo(false);
  return true;
}

void SubSequenceLayer::forward(PassType passType) {
  Layer::forward(passType);

  size_t dim = getSize();

  const Argument& input = getInput(0);
  size_t numSequences1 = input.getNumSequences();
  auto startPositions1 = input.sequenceStartPositions->getVector(false);

  const Argument& offsetSeq = getInput(1);
  size_t numSequences2 = offsetSeq.getNumSequences();
  auto startPositions2 = offsetSeq.sequenceStartPositions->getVector(false);

  const Argument& sizeSeq = getInput(2);
  size_t numSequences3 = sizeSeq.getNumSequences();
  auto startPositions3 = sizeSeq.sequenceStartPositions->getVector(false);

  CHECK_EQ(dim, input.value->getWidth());

  CHECK_EQ(startPositions1->getData()[numSequences1], input.getBatchSize());
  CHECK_EQ(numSequences1, startPositions1->getSize() - 1);

  CHECK_EQ(startPositions2->getData()[numSequences2], offsetSeq.getBatchSize());
  CHECK_EQ(numSequences2, startPositions2->getSize() - 1);

  CHECK_EQ(startPositions3->getData()[numSequences3], sizeSeq.getBatchSize());
  CHECK_EQ(numSequences3, startPositions3->getSize() - 1);

  CHECK_EQ(numSequences1, numSequences2);
  CHECK_EQ(numSequences2, numSequences3);

  MatrixPtr inputValue = input.value;
  IVectorPtr offsetValue;
  IVectorPtr sizeValue;

  if (useGpu_) {
    // copy to cpu
    IVector::resizeOrCreate(offsetValue, offsetSeq.ids->getSize(), false);
    IVector::resizeOrCreate(sizeValue, sizeSeq.ids->getSize(), false);
    offsetValue->copyFrom(*offsetSeq.ids);
    sizeValue->copyFrom(*sizeSeq.ids);
  } else {
    offsetValue = offsetSeq.ids;
    sizeValue = sizeSeq.ids;
  }

  CHECK_EQ(offsetValue->getSize(), numSequences1);
  CHECK_EQ(sizeValue->getSize(), numSequences1);

  int* offsets = offsetValue->getData();
  int* sizes = sizeValue->getData();

  // get total height of output
  size_t height = 0;
  for (size_t seqId = 0; seqId < numSequences1; seqId++) {
    height += sizes[seqId];
  }

  // reset output
  resetOutput(height, dim);

  MatrixPtr outputValue = getOutputValue();

  const int* starts1 = startPositions1->getData();

  {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SubSequenceLayerForward", getName().c_str());

    size_t offsetIn = 0;
    size_t offsetOut = 0;
    size_t size = 0;
    for (size_t seqId = 0; seqId < numSequences1; ++seqId) {
      offsetIn = starts1[seqId] + offsets[seqId];
      size = sizes[seqId];

      outputValue->subMatrix(offsetOut, size, tmpDest_)
          ->assign(*(inputValue->subMatrix(offsetIn, size, tmpSrc_)));

      offsetOut += size;
    }

    // modify the sequenceStartPositions
    ICpuGpuVector::resizeOrCreate(
        output_.sequenceStartPositions, numSequences1 + 1, false);

    int* tgtBuf = output_.sequenceStartPositions->getMutableData(false);
    int offset = 0;
    for (size_t seqId = 0; seqId < numSequences1; ++seqId) {
      tgtBuf[seqId] = offset;
      offset += sizes[seqId];
    }
    tgtBuf[numSequences1] = offset;
  }

  if (biases_.get() != NULL) {
    MatrixPtr outV = getOutputValue();
    outV->addBias(*(biases_->getW()), 1);
  }

  /* activation */
  forwardActivation();
}

void SubSequenceLayer::backward(const UpdateCallback& callback) {
  /* activation */
  backwardActivation();

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    // Increasing the number of gradient
    biases_->getParameterPtr()->incUpdate(callback);
  }

  MatrixPtr inputGrad1 = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();
  auto startPositions1 = getInput(0).sequenceStartPositions->getVector(false);
  size_t numSequences1 = startPositions1->getSize() - 1;
  const int* starts1 = startPositions1->getData();

  const Argument& offsetSeq = getInput(1);
  const Argument& sizeSeq = getInput(2);
  IVectorPtr offsetValue;
  IVectorPtr sizeValue;

  if (useGpu_) {
    // copy to cpu
    IVector::resizeOrCreate(offsetValue, offsetSeq.ids->getSize(), false);
    IVector::resizeOrCreate(sizeValue, sizeSeq.ids->getSize(), false);
    offsetValue->copyFrom(*offsetSeq.ids);
    sizeValue->copyFrom(*sizeSeq.ids);
  } else {
    offsetValue = offsetSeq.ids;
    sizeValue = sizeSeq.ids;
  }

  int* offsets = offsetValue->getData();
  int* sizes = sizeValue->getData();
  {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SubSequenceLayerBackward", getName().c_str());

    int offsetIn = 0;
    int offsetOut = 0;
    int size = 0;
    for (size_t seqId = 0; seqId < numSequences1; ++seqId) {
      offsetIn = starts1[seqId] + offsets[seqId];
      size = sizes[seqId];

      inputGrad1->subMatrix(offsetIn, size, tmpDest_)
          ->add(*(outputGrad->subMatrix(offsetOut, size, tmpSrc_)));
      offsetOut += size;
    }
  }
}

}  // namespace paddle
