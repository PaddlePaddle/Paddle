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

#include "CTCLayer.h"

/* Please reference the Chapter7  in
 * "Alex graves, Supervised Sequence Labelling with
 * Recurrent Neural Networks" */
namespace paddle {
REGISTER_LAYER(ctc, CTCLayer);

bool CTCLayer::init(const LayerMap& layerMap,
                    const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2UL);

  /* The inputLayers_[0] must be softmax output */
  numClasses_ = inputLayers_[0]->getSize();
  normByTimes_ = config_.norm_by_times();
  CHECK_GE(numClasses_, 2UL);

  // We don't need sequenceStartPositions because each sample of output_ is
  // for the cost of one sequence.
  setNeedSequenceInfo(false);
  if (useGpu_) {
    tmpCpuInput_.reserve(inputLayers_.size());
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      tmpCpuInput_.push_back(Argument());
    }
  }
  return true;
}

void CTCLayer::forward(PassType passType) {
  Layer::forward(passType);
  if (useGpu_) {
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      tmpCpuInput_[i].resizeAndCopyFrom(
          getInput(i), false, HPPL_STREAM_DEFAULT);
    }
    hl_stream_synchronize(HPPL_STREAM_DEFAULT);
    forwardImp(tmpCpuInput_[0], tmpCpuInput_[1]);
  } else {
    forwardImp(getInput(0), getInput(1));
  }
}

void CTCLayer::forwardImp(const Argument& softmaxSeqs,
                          const Argument& labelSeqs) {
  CHECK(softmaxSeqs.sequenceStartPositions);
  CHECK(labelSeqs.sequenceStartPositions);
  CHECK(labelSeqs.ids);

  size_t numSequences = labelSeqs.sequenceStartPositions->getSize() - 1;
  CHECK_EQ(numSequences, softmaxSeqs.sequenceStartPositions->getSize() - 1);

  resizeOutput(numSequences, 1);
  std::vector<real> out(numSequences);

  const int* labelSeqsStarts = labelSeqs.sequenceStartPositions->getData(false);
  const int* softmaxSeqsStarts =
      softmaxSeqs.sequenceStartPositions->getData(false);

  for (size_t i = 0; i < numSequences; i++) {
    if (i >= ctcs_.size()) {
      ctcs_.emplace_back(numClasses_, normByTimes_);
    }
    out[i] = ctcs_[i].forward(
        softmaxSeqs.value->getData() + numClasses_ * softmaxSeqsStarts[i],
        softmaxSeqsStarts[i + 1] - softmaxSeqsStarts[i],
        labelSeqs.ids->getData() + labelSeqsStarts[i],
        labelSeqsStarts[i + 1] - labelSeqsStarts[i]);
  }
  output_.value->copyFrom(out.data(), numSequences);
}

void CTCLayer::backward(const UpdateCallback& callback) {
  (void)callback;
  if (useGpu_) {
    backwardImp(callback, tmpCpuInput_[0], tmpCpuInput_[1]);
    const_cast<Argument&>(getInput(0))
        .resizeAndCopyFrom(tmpCpuInput_[0], true, HPPL_STREAM_DEFAULT);
    const_cast<Argument&>(getInput(1))
        .resizeAndCopyFrom(tmpCpuInput_[1], true, HPPL_STREAM_DEFAULT);
  } else {
    backwardImp(callback, getInput(0), getInput(1));
  }
}

void CTCLayer::backwardImp(const UpdateCallback& callback,
                           const Argument& softmaxSeqs,
                           const Argument& labelSeqs) {
  size_t numSequences = labelSeqs.sequenceStartPositions->getSize() - 1;

  const int* labelSeqsStarts = labelSeqs.sequenceStartPositions->getData(false);
  const int* softmaxSeqsStarts =
      softmaxSeqs.sequenceStartPositions->getData(false);

  for (size_t i = 0; i < numSequences; ++i) {
    ctcs_[i].backward(
        softmaxSeqs.value->getData() + numClasses_ * softmaxSeqsStarts[i],
        softmaxSeqs.grad->getData() + numClasses_ * softmaxSeqsStarts[i],
        labelSeqs.ids->getData() + labelSeqsStarts[i],
        labelSeqsStarts[i + 1] - labelSeqsStarts[i]);
  }
}

}  // namespace paddle
