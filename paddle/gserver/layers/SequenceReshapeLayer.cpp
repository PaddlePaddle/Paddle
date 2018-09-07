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
 *  A layer for reshaping the sequence. Assume the input sequence has
 *  T instances, the dimension of each instance is M, and the input
 *  reshape_dim is N, then the output sequence has T*M/N instances,
 *  the dimension of each instance is N.
 *
 *  Note that T*M/N must be an integer.
 */

class SequenceReshapeLayer : public Layer {
 protected:
  std::unique_ptr<Weight> biases_;

  MatrixPtr reshapedOutputGrad;

 public:
  explicit SequenceReshapeLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(seqreshape, SequenceReshapeLayer);

bool SequenceReshapeLayer::init(const LayerMap& layerMap,
                                const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(1U, inputLayers_.size());

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }
  setNeedSequenceInfo(false);
  return true;
}

void SequenceReshapeLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& input = getInput(0);

  size_t inDim = input.value->getWidth();
  size_t outDim = getSize();

  size_t numSequences = input.getNumSequences();

  // by default, we assume each instance as a sequence
  IVectorPtr seqStarts;
  IVector::resizeOrCreate(seqStarts, input.getBatchSize() + 1, false);
  int* startsData = seqStarts->getData();
  for (int i = 0; i < input.getBatchSize() + 1; i++) {
    startsData[i] = i;
  }
  const int* starts = startsData;

  // if there is sequence, then use start positions
  if (input.sequenceStartPositions) {
    auto startPositions = input.sequenceStartPositions->getVector(false);
    starts = startPositions->getData();
    CHECK_EQ(starts[numSequences], input.getBatchSize());
    CHECK_EQ(numSequences, startPositions->getSize() - 1);
  }

  for (size_t seqID = 0; seqID < numSequences; seqID++) {
    size_t inNumIns = starts[seqID + 1] - starts[seqID];
    size_t outNumIns = inNumIns * inDim / outDim;
    CHECK_EQ(outNumIns * outDim, inNumIns * inDim);
  }

  MatrixPtr inputValue = getInputValue(0);

  // reset output
  reserveOutput(inputValue->getHeight() * inDim / outDim, outDim);
  MatrixPtr outputValue = getOutputValue();

  {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SequenceReshapeLayerForward", getName().c_str());

    outputValue->copyFrom(*inputValue);

    // modify the sequenceStartPositions
    ICpuGpuVector::resizeOrCreate(
        output_.sequenceStartPositions, numSequences + 1, false);

    int* tgtBuf = output_.sequenceStartPositions->getMutableData(false);

    for (size_t seqId = 0; seqId < numSequences + 1; ++seqId) {
      tgtBuf[seqId] = starts[seqId] * inDim / outDim;
    }
  }

  if (biases_.get() != NULL) {
    MatrixPtr outV = getOutputValue();
    outV->addBias(*(biases_->getW()), 1);
  }

  /* activation */
  forwardActivation();
}

void SequenceReshapeLayer::backward(const UpdateCallback& callback) {
  /* activation */
  backwardActivation();

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    // Increasing the number of gradient
    biases_->getParameterPtr()->incUpdate(callback);
  }

  MatrixPtr inputGrad = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();

  AsyncGpuBlock asyncGpuBlock;
  REGISTER_TIMER_INFO("SequenceReshapeLayerBackward", getName().c_str());

  if (inputGrad) {
    Matrix::resizeOrCreate(reshapedOutputGrad,
                           inputGrad->getHeight(),
                           inputGrad->getWidth(),
                           false,
                           useGpu_);
    reshapedOutputGrad->copyFrom(*outputGrad);
    inputGrad->add(*reshapedOutputGrad);
  }
}

}  // namespace paddle
