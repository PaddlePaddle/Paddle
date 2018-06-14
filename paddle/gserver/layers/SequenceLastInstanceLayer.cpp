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

#include "paddle/utils/Logging.h"

#include "SequencePoolLayer.h"
#include "paddle/math/Matrix.h"
#include "paddle/utils/Stat.h"

namespace paddle {

/**
 * A layer for extracting the last instance of the input sequence.
 * Input: a sequence
 * If SequenceLevel = kNonseq:
 *   Output: a sequence containing only the last instance of the input sequence
 *   If stride_ > 0:
 *      Output: a shorten sequence. Stride is the step size by which we slide a
 *              window upon the input sequence, and getting last instance
 *              operation is then applied to each interval independently.
 * If SequenceLevel = kSeq:
 *   Check input sequence must has sub-sequence
 *   Output: a sequence containing only the last instance of each sub-sequence
 *           of the input sequence
 *
 * The config file api is last_seq and first_seq.
 */

class SequenceLastInstanceLayer : public SequencePoolLayer {
 protected:
  MatrixPtr tmpSrc_;
  MatrixPtr tmpDest_;
  std::vector<int> instanceIds_;

 public:
  explicit SequenceLastInstanceLayer(const LayerConfig& config)
      : SequencePoolLayer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;
};

REGISTER_LAYER(seqlastins, SequenceLastInstanceLayer);

bool SequenceLastInstanceLayer::init(const LayerMap& layerMap,
                                     const ParameterMap& parameterMap) {
  SequencePoolLayer::init(layerMap, parameterMap);
  reversed_ = config_.select_first();

  tmpSrc_ =
      Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, useGpu_);
  tmpDest_ =
      Matrix::create(nullptr, /* height= */ 1, 1, /* trans= */ false, useGpu_);

  return true;
}

void SequenceLastInstanceLayer::forward(PassType passType) {
  SequencePoolLayer::forward(passType);

  auto starts = startPositions_->getData(false);
  MatrixPtr inputValue = getInputValue(0);
  MatrixPtr outputValue = getOutputValue();

  {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SequenceLastInstanceLayerForward", getName().c_str());

    instanceIds_.clear();
    for (size_t seqId = 0; seqId < newBatchSize_; ++seqId) {
      int insId = reversed_ ? starts[seqId] : starts[seqId + 1] - 1;
      instanceIds_.push_back(insId);

      outputValue->subMatrix(seqId, 1, tmpDest_)
          ->assign(*(inputValue->subMatrix(insId, 1, tmpSrc_)));
    }
  }

  if (biases_.get() != NULL) {
    outputValue->addBias(*(biases_->getW()), 1);
  }

  /*  activation, should set to 'linear' in most cases */
  forwardActivation();
}

void SequenceLastInstanceLayer::backward(const UpdateCallback& callback) {
  SequencePoolLayer::backward(callback);

  MatrixPtr inputGrad = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();

  if (inputGrad) {
    AsyncGpuBlock asyncGpuBlock;
    REGISTER_TIMER_INFO("SequenceLastInstanceLayerBackward", getName().c_str());

    for (size_t seqId = 0; seqId < newBatchSize_; ++seqId) {
      inputGrad->subMatrix(instanceIds_[seqId], 1, tmpDest_)
          ->add(*(outputGrad->subMatrix(seqId, 1, tmpSrc_)));
    }
  }
}

}  // namespace paddle
