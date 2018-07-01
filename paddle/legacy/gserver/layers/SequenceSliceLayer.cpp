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

class SequenceSliceLayer : public Layer {
 public:
  explicit SequenceSliceLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 private:
  /*
   * TODO(caoying)
   * In PaddePaddle, currently all matrices are real number types,
   * but the second and the (optional) third input which are some
   * selected indices of the give sequence to trim the sequence, are actually
   * filled with int types so that storing int types information in real number
   * matrices is very dangerous, since real numbers will be convered to int
   * types. If a user fills this matrix himself, invalid data may occor.
   */

  MatrixPtr startIdsOnCpu_;
  MatrixPtr endIdsOnCpu_;

  std::vector<int> selectedRows_;
  IVectorPtr rowIndice_;
  std::vector<std::vector<int>> inputSeqInfoVec_;
  std::vector<int> outSubSeqStartPos_;
  std::vector<int> outSeqStartPos_;

  void checkInputs();
  void copySliceIdsToCpu();
  void calSelectedRows(const MatrixPtr starts, const MatrixPtr ends);
};

REGISTER_LAYER(seq_slice, SequenceSliceLayer);

bool SequenceSliceLayer::init(const LayerMap& layerMap,
                              const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);
  CHECK_GE(inputLayers_.size(), 2U);
  CHECK_LE(inputLayers_.size(), 3U);

  setNeedSequenceInfo(false);
  return true;
}

void SequenceSliceLayer::checkInputs() {
  const Argument& inputSeq = getInput(0);
  CHECK(inputSeq.hasSeq()) << "The first input of sequence slice layer "
                           << "must be a sequence.";
  const MatrixPtr indices1 = getInputValue(1);
  CHECK_EQ(
      indices1->getHeight(),
      static_cast<size_t>(inputSeq.hasSubseq() ? inputSeq.getNumSubSequences()
                                               : inputSeq.getNumSequences()))
      << "Height of the second input should be equal to number of sequence "
      << "in the first input.";
  if (inputLayers_.size() == 3) {
    const MatrixPtr indices2 = getInputValue(2);
    CHECK_EQ(indices2->getHeight(), indices1->getHeight())
        << "start indices and end indices should have the same height.";
    CHECK_EQ(indices2->getWidth(), indices1->getWidth())
        << "start indices and end indices should have the same Width.";
  }
}

void SequenceSliceLayer::copySliceIdsToCpu() {
  const MatrixPtr indices1 = getInputValue(1);
  if (inputLayers_.size() == 2U) {
    if (config_.select_first()) {
      Matrix::resizeOrCreate(startIdsOnCpu_,
                             indices1->getHeight(),
                             indices1->getWidth(),
                             false /* trans */,
                             false /* useGpu */);
      startIdsOnCpu_->copyFrom(*indices1);
      endIdsOnCpu_ = nullptr;
    } else {
      Matrix::resizeOrCreate(endIdsOnCpu_,
                             indices1->getHeight(),
                             indices1->getWidth(),
                             false /* trans */,
                             false /* useGpu */);
      endIdsOnCpu_->copyFrom(*indices1);
      startIdsOnCpu_ = nullptr;
    }
  } else if (inputLayers_.size() == 3U) {
    Matrix::resizeOrCreate(startIdsOnCpu_,
                           indices1->getHeight(),
                           indices1->getWidth(),
                           false /* trans */,
                           false /* useGpu */);
    startIdsOnCpu_->copyFrom(*indices1);

    const MatrixPtr indices2 = getInputValue(2);
    Matrix::resizeOrCreate(endIdsOnCpu_,
                           indices2->getHeight(),
                           indices2->getWidth(),
                           false /* trans */,
                           false /* useGpu */);
    endIdsOnCpu_->copyFrom(*indices2);
  }
}

void SequenceSliceLayer::calSelectedRows(const MatrixPtr starts,
                                         const MatrixPtr ends) {
  CHECK(starts || ends) << "At least one of the start or end indices "
                        << "should be given.";

  bool hasSubseq = getInput(0).hasSubseq();

  outSeqStartPos_.resize(1, 0);
  outSubSeqStartPos_.resize(1, 0);
  selectedRows_.clear();

  size_t beamSize = starts ? starts->getWidth() : ends->getWidth();
  size_t rowIdx = 0;
  for (size_t i = 0; i < inputSeqInfoVec_.size(); ++i) {
    for (size_t j = 0; j < inputSeqInfoVec_[i].size() - 1; ++j) {
      for (size_t k = 0; k < beamSize; ++k) {
        if (starts && starts->getElement(rowIdx, k) == -1.) break;
        if (ends && ends->getElement(rowIdx, k) == -1.) break;

        int begPos = inputSeqInfoVec_[i][j];
        if (starts) begPos += starts->getElement(rowIdx, k);

        int endPos = inputSeqInfoVec_[i][j + 1] - 1;
        if (ends) endPos = inputSeqInfoVec_[i][j] + ends->getElement(rowIdx, k);

        int seqLen = endPos - begPos + 1;
        CHECK_GT(seqLen, 0);
        for (int m = begPos; m <= endPos; ++m) selectedRows_.push_back(m);
        hasSubseq
            ? outSubSeqStartPos_.push_back(outSubSeqStartPos_.back() + seqLen)
            : outSeqStartPos_.push_back(outSeqStartPos_.back() + seqLen);
      }
      rowIdx++;
    }
    if (hasSubseq) outSeqStartPos_.push_back(outSubSeqStartPos_.back());
  }

  if (useGpu_) {
    rowIndice_ = IVector::create(selectedRows_.size(), useGpu_);
    rowIndice_->copyFrom(selectedRows_.data(), selectedRows_.size());
  } else {
    rowIndice_ =
        IVector::create(selectedRows_.data(), selectedRows_.size(), useGpu_);
  }

  // create the sequence information for the output.
  ICpuGpuVector::resizeOrCreate(
      output_.sequenceStartPositions, outSeqStartPos_.size(), false);
  output_.sequenceStartPositions->copyFrom(
      outSeqStartPos_.data(), outSeqStartPos_.size(), false);

  if (hasSubseq) {
    ICpuGpuVector::resizeOrCreate(
        output_.subSequenceStartPositions, outSubSeqStartPos_.size(), false);
    output_.subSequenceStartPositions->copyFrom(
        outSubSeqStartPos_.data(), outSubSeqStartPos_.size(), false);
  }
}

void SequenceSliceLayer::forward(PassType passType) {
  Layer::forward(passType);
  checkInputs();

  const Argument& inputSeq = getInput(0);
  inputSeqInfoVec_.clear();
  Argument::reorganizeSeqInfo(inputSeq.sequenceStartPositions,
                              inputSeq.subSequenceStartPositions,
                              inputSeqInfoVec_);
  if (!useGpu_) {
    if (inputLayers_.size() == 2U) {
      startIdsOnCpu_ = config_.select_first() ? getInputValue(1) : nullptr;
      endIdsOnCpu_ = config_.select_first() ? nullptr : getInputValue(1);
    } else if (inputLayers_.size() == 3U) {
      startIdsOnCpu_ = getInputValue(1);
      endIdsOnCpu_ = getInputValue(2);
    }
  } else {
    copySliceIdsToCpu();
  }

  /*
   * calculate the selected row indices in a batch, and build the output
   * sequence information.
   */
  calSelectedRows(startIdsOnCpu_, endIdsOnCpu_);

  resetOutput(selectedRows_.size(), getSize());

  getOutputValue()->selectRows(*getInputValue(0), *rowIndice_);
}

void SequenceSliceLayer::backward(const UpdateCallback& callback) {
  getOutputGrad()->addToRows(*getInputGrad(0), *rowIndice_);
}

}  // namespace paddle
