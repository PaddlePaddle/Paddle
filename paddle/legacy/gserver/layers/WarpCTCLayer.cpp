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

#include "WarpCTCLayer.h"

namespace paddle {

REGISTER_LAYER(warp_ctc, WarpCTCLayer);

bool WarpCTCLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  /* Initialize the basic parament class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2UL);

  /* The inputLayers_[0] must be sequence output without softmax */
  numClasses_ = config_.size();
  CHECK_GE(numClasses_, 2UL);
  CHECK_EQ(numClasses_, inputLayers_[0]->getSize());

  blank_ = config_.blank();
  CHECK_LT(blank_, numClasses_);

  normByTimes_ = config_.norm_by_times();

  // We don't need sequenceStartPositions because each sample of output_ is
  // for the cost of one sequence.
  setNeedSequenceInfo(false);

  return true;
}

void WarpCTCLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& output = getInput(0);
  const Argument& labels = getInput(1);

  CHECK(output.sequenceStartPositions);
  CHECK(labels.sequenceStartPositions);
  CHECK(labels.ids);

  size_t numSequences = labels.sequenceStartPositions->getSize() - 1;
  CHECK_EQ(numSequences, output.sequenceStartPositions->getSize() - 1);

  resizeOutput(numSequences, 1);

  const int* cpuLabelStartPositions =
      labels.sequenceStartPositions->getData(false);
  const int* cpuOutputStartPositions =
      output.sequenceStartPositions->getData(false);

  std::vector<int> cpuLabelLengths(numSequences);
  std::vector<int> cpuOutputLengths(numSequences);
  for (size_t i = 0; i < numSequences; i++) {
    cpuLabelLengths[i] =
        cpuLabelStartPositions[i + 1] - cpuLabelStartPositions[i];
    cpuOutputLengths[i] =
        cpuOutputStartPositions[i + 1] - cpuOutputStartPositions[i];
  }

  /* Get the maximum sequence length */
  maxSequenceLength_ = 0;
  maxSequenceLength_ = *std::max_element(
      cpuOutputLengths.data(), cpuOutputLengths.data() + numSequences);

  Matrix::resizeOrCreate(batchValue_,
                         /* height */ numSequences * maxSequenceLength_,
                         /* width */ numClasses_,
                         /* trans */ false,
                         /* useGpu */ useGpu_);

  Matrix::resizeOrCreate(batchGrad_,
                         /* height */ numSequences * maxSequenceLength_,
                         /* width */ numClasses_,
                         /* trans */ false,
                         /* useGpu */ useGpu_);
  batchGrad_->zeroMem();

  seq2batchPadding(output.value, batchValue_, output.sequenceStartPositions);

  /* labels always in CPU memory */
  IVector::resizeOrCreate(cpuLabels_,
                          /* size */ (labels.ids)->getSize(),
                          /* useGpu */ false);
  cpuLabels_->copyFrom(*(labels.ids));

  /* labels always in CPU memory */
  Matrix::resizeOrCreate(cpuCosts_,
                         /* height */ numSequences,
                         /* width */ 1,
                         /* trans */ false,
                         /* useGpu */ false);

  /* Init warp-ctc options */
  hl_warpctc_options_t options;
  hl_warpctc_init(blank_, useGpu_, &options);

  /* Get the needed workspace size */
  size_t workspaceBytes = 0;
  hl_warpctc_get_workspace_size(cpuLabelLengths.data(),
                                cpuOutputLengths.data(),
                                numClasses_,
                                numSequences,
                                &options,
                                &workspaceBytes);
  CHECK_GT(workspaceBytes, 0UL);

  size_t workspaceLength = workspaceBytes / sizeof(real) + 1;
  Vector::resizeOrCreate(workspace_,
                         /* size */ workspaceLength,
                         /* useGpu */ useGpu_);

  hl_warpctc_compute_loss(batchValue_->getData(),
                          batchGrad_->getData(),
                          cpuLabels_->getData(),
                          cpuLabelLengths.data(),
                          cpuOutputLengths.data(),
                          numClasses_,
                          numSequences,
                          cpuCosts_->getData(),
                          workspace_->getData(),
                          &options);

  /* Copy the costs */
  output_.value->copyFrom(*cpuCosts_);
}

void WarpCTCLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  const Argument& output = getInput(0);
  CHECK(batchGrad_);

  batch2seqPadding(
      output.grad, batchGrad_, output.sequenceStartPositions, normByTimes_);
}

void WarpCTCLayer::seq2batchPadding(const MatrixPtr& seqValue,
                                    MatrixPtr& batchValue,
                                    const ICpuGpuVectorPtr& seqStartPositions) {
  size_t numSequences = seqStartPositions->getSize() - 1;
  const int* seqStartPositionsData = seqStartPositions->getData(useGpu_);

  real* seqData = seqValue->getData();
  real* batchData = batchValue->getData();
  if (useGpu_) {
    hl_sequence2batch_copy_padding(batchData,
                                   seqData,
                                   seqStartPositionsData,
                                   numClasses_,
                                   maxSequenceLength_,
                                   numSequences,
                                   false,
                                   true);
  } else {
    for (size_t i = 0; i < maxSequenceLength_; i++) {
      for (size_t j = 0; j < numSequences; j++) {
        size_t sequenceStart = seqStartPositionsData[j];
        size_t sequenceLength =
            seqStartPositionsData[j + 1] - seqStartPositionsData[j];
        if (i < sequenceLength) {
          memcpy(batchData + (i * numSequences + j) * numClasses_,
                 seqData + (sequenceStart + i) * numClasses_,
                 numClasses_ * sizeof(real));
        } else {
          memset(batchData + (i * numSequences + j) * numClasses_,
                 0,
                 numClasses_ * sizeof(real));
        }
      }
    }
  }
}

void WarpCTCLayer::batch2seqPadding(const MatrixPtr& seqValue,
                                    MatrixPtr& batchValue,
                                    const ICpuGpuVectorPtr& seqStartPositions,
                                    bool normByTimes) {
  size_t numSequences = seqStartPositions->getSize() - 1;
  const int* seqStartPositionsData = seqStartPositions->getData(useGpu_);

  real* seqData = seqValue->getData();
  real* batchData = batchValue->getData();
  if (useGpu_) {
    hl_sequence2batch_copy_padding(batchData,
                                   seqData,
                                   seqStartPositionsData,
                                   numClasses_,
                                   maxSequenceLength_,
                                   numSequences,
                                   normByTimes,
                                   false);
  } else {
    for (size_t i = 0; i < numSequences; i++) {
      int sequenceStart = seqStartPositionsData[i];
      int sequenceLength =
          seqStartPositionsData[i + 1] - seqStartPositionsData[i];
      real scale = normByTimes ? (1.0f / (real)sequenceLength) : 1.0f;
      for (int j = 0; j < sequenceLength; j++) {
        for (size_t k = 0; k < numClasses_; k++) {
          seqData[(sequenceStart + j) * numClasses_ + k] =
              batchData[(j * numSequences + i) * numClasses_ + k] * scale;
        }
      }
    }
  }
}

}  // namespace paddle
