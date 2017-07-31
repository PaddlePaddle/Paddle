/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

class SubNestedSequenceLayer : public Layer {
public:
  explicit SubNestedSequenceLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

private:
  void checkInputs(const Argument& inputSeq, const Argument& seqScores);
  void calSelectedCols(const Argument& scores,
                       const int* subSeqStartPos,
                       size_t topK);
  void partialSortIndex(const std::vector<real>& values,
                        int k,
                        std::vector<size_t>& indices);
  void buildOutputSeqInfo();

  std::vector<int> outSeqStartInfo_;
  std::vector<int> outSubSeqStartInfo_;

  MatrixPtr scoreOverInputSeq_;

  // rowIdx_ and selectedRows_ actually share a same memory.
  IVectorPtr rowIndice_;
  std::vector<int> selectedRows_;
};

REGISTER_LAYER(sub_nested_seq, SubNestedSequenceLayer);

bool SubNestedSequenceLayer::init(const LayerMap& layerMap,
                                  const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(2U, inputLayers_.size());
  setNeedSequenceInfo(false);
  return true;
}

void SubNestedSequenceLayer::checkInputs(const Argument& inputSeq,
                                         const Argument& seqScores) {
  CHECK(inputSeq.hasSubseq()) << "The first input of SubNestSequence layer "
                              << "must be a nested sequence.";
  CHECK(seqScores.hasSeq())
      << "The second input of SubNestSequence layer must be a sequence.";
  CHECK_EQ(seqScores.value->getWidth(), 1U)
      << "The second input of SubNestedSequenceLayer is scores "
      << "over each sequence in a nested sequence, "
      << "so its size should be 1.";
  CHECK_EQ(inputSeq.getNumSubSequences(), seqScores.value->getHeight())
      << "The second input of SubNestedSequenceLayer is scores "
      << "over each sequence in a nested sequence, so its height should be "
      << "equal to number of sequence in the first input.";
}

void SubNestedSequenceLayer::partialSortIndex(const std::vector<real>& values,
                                              int k,
                                              std::vector<size_t>& indices) {
  CHECK_GE(values.size(), k);
  indices.resize(values.size(), 0);
  std::iota(begin(indices), end(indices), 0U);
  std::partial_sort(begin(indices),
                    begin(indices) + k,
                    end(indices),
                    [&](size_t a, size_t b) { return values[a] > values[b]; });
}

void SubNestedSequenceLayer::calSelectedCols(const Argument& scores,
                                             const int* subSeqStartPos,
                                             size_t topK) {
  selectedRows_.clear();
  outSubSeqStartInfo_.resize(1, 0);
  outSeqStartInfo_.resize(1, 0);

  real* seqScores = nullptr;
  if (useGpu_) {
    Matrix::resizeOrCreate(scoreOverInputSeq_,
                           scores.value->getHeight(),
                           scores.value->getWidth(),
                           false /* trans */,
                           false /* useGpu */);
    scoreOverInputSeq_->copyFrom(*scores.value);
    seqScores = scoreOverInputSeq_->getData();
  } else {
    seqScores = scores.value->getData();
  }

  int* scoreSeqStartPos = scores.sequenceStartPositions->getMutableData(false);
  for (int i = 0; i < scores.getNumSequences(); ++i) {
    int seqLen = scoreSeqStartPos[i + 1] - scoreSeqStartPos[i];
    int selectedSeqNum = std::min(static_cast<int>(config_.top_k()), seqLen);

    std::vector<size_t> sortedIdx;
    partialSortIndex(std::vector<real>(seqScores + scoreSeqStartPos[i],
                                       seqScores + scoreSeqStartPos[i + 1]),
                     selectedSeqNum,
                     sortedIdx);

    for (int j = 0; j < selectedSeqNum; ++j) {
      int begPos = subSeqStartPos[scoreSeqStartPos[i] + sortedIdx[j]];
      int endPos = subSeqStartPos[scoreSeqStartPos[i] + sortedIdx[j] + 1];
      for (int m = begPos; m < endPos; ++m) selectedRows_.push_back(m);
      outSubSeqStartInfo_.push_back(outSubSeqStartInfo_.back() + endPos -
                                    begPos);
    }
    outSeqStartInfo_.push_back(outSubSeqStartInfo_.back());
  }
}

void SubNestedSequenceLayer::buildOutputSeqInfo() {
  Argument& output = getOutput();

  ICpuGpuVector::resizeOrCreate(
      output.sequenceStartPositions, outSeqStartInfo_.size(), false);
  output.sequenceStartPositions->copyFrom(
      outSeqStartInfo_.data(), outSeqStartInfo_.size(), false);

  ICpuGpuVector::resizeOrCreate(
      output.subSequenceStartPositions, outSubSeqStartInfo_.size(), false);
  output.subSequenceStartPositions->copyFrom(
      outSubSeqStartInfo_.data(), outSubSeqStartInfo_.size(), false);
}

void SubNestedSequenceLayer::forward(PassType passType) {
  Layer::forward(passType);
  const Argument& inputSeq = getInput(0);
  const Argument& seqScores = getInput(1);

  checkInputs(inputSeq, seqScores);

  calSelectedCols(seqScores,
                  inputSeq.subSequenceStartPositions->getMutableData(false),
                  config_.top_k());
  resetOutput(selectedRows_.size(), getSize());
  buildOutputSeqInfo();

  if (useGpu_) {
    rowIndice_ = IVector::create(selectedRows_.size(), useGpu_);
    rowIndice_->copyFrom(selectedRows_.data(), selectedRows_.size());
  } else {
    rowIndice_ =
        IVector::create(selectedRows_.data(), selectedRows_.size(), useGpu_);
  }

  getOutputValue()->selectRows(*getInputValue(0), *rowIndice_);
}

void SubNestedSequenceLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inputGrad1 = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();

  if (inputGrad1) outputGrad->addToRows(*inputGrad1, *rowIndice_);
}

}  // namespace paddle
