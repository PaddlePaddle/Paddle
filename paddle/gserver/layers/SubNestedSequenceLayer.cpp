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

class SubNestedSequenceLayer : public Layer {
 public:
  explicit SubNestedSequenceLayer(const LayerConfig& config) : Layer(config) {}

  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;

  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback = nullptr) override;

 private:
  /*
   * This functions generates the indices of rows in a batch according to the
   * indices of selected sub-sequence in each sequence.
   *
   * Examples:
   * selectedIndices:
   *   [
   *     [0, 1, -1],
   *     [0, 1, 2],
   *     [0, -1, -1],
   *     [0, 2, 3],
   *   ]
   * inputSeqInfo:
   *   [
   *     [0,3,4],
   *     [4,5,7,10,15],
   *     [15,20],
   *     [20,22,23,25,28]
   *   ]
   *
   * ths output is saved to private member rowIndice_;
   * [0,1,2,3,4,5,6,7,8,9,15,16,17,18,19,20,21,23,24,25,26,27]
   */

  void calSelectedRows(const MatrixPtr selectedIndices,
                       const std::vector<std::vector<int>>& inputSeqInfo);

  /*
   * TODO(caoying)
   * In PaddePaddle, currently all matrices are real number types,
   * but the second is some selected indices of the give sequence to trim
   * the nested sequence, are actually filled with int types so that storing
   * int types information in real number matrices is very dangerous, since
   * real numbers will be convered to int types. If a user fills this matrix
   * himself, invalid data may occor.
   *
   * if the second input of this layer is on GPU memory, copy it to CPU memory.
   */
  MatrixPtr selIdsCpu_;

  /*
   * reorganize sequenceStartPositions and subSequenceStartPositions
   * into a 2d vector to facilitate the sequence selection process.
   */
  std::vector<std::vector<int>> inputSeqInfoVec_;

  /* store the final selected row indices in a batch */
  IVectorPtr rowIndice_;
  /* rowIndice_ and selectedRows_ actually share a same memory. */
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

void SubNestedSequenceLayer::calSelectedRows(
    const MatrixPtr selectedIndices,
    const std::vector<std::vector<int>>& inputSeqInfo) {
  selectedRows_.clear();

  std::vector<int> outSeqStartInfo(1, 0);
  std::vector<int> outSubSeqStartInfo(1, 0);

  size_t seqNum = selectedIndices->getHeight();
  size_t beamSize = selectedIndices->getWidth();
  for (size_t i = 0; i < seqNum; ++i) {
    for (size_t j = 0; j < beamSize; ++j) {
      if (selectedIndices->getElement(i, j) == -1.) break;
      size_t selSubSeqIdx = selectedIndices->getElement(i, j);
      CHECK_GT(inputSeqInfoVec_[i].size() - 1, selSubSeqIdx);

      size_t subSeqLen = inputSeqInfoVec_[i][selSubSeqIdx + 1] -
                         inputSeqInfoVec_[i][selSubSeqIdx];
      for (size_t k = 0; k < subSeqLen; ++k)
        selectedRows_.push_back(inputSeqInfoVec_[i][selSubSeqIdx] + k);
      outSubSeqStartInfo.push_back(outSubSeqStartInfo.back() + subSeqLen);
    }
    outSeqStartInfo.push_back(outSubSeqStartInfo.back());
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
      output_.sequenceStartPositions, outSeqStartInfo.size(), false);
  output_.sequenceStartPositions->copyFrom(
      outSeqStartInfo.data(), outSeqStartInfo.size(), false);

  ICpuGpuVector::resizeOrCreate(
      output_.subSequenceStartPositions, outSubSeqStartInfo.size(), false);
  output_.subSequenceStartPositions->copyFrom(
      outSubSeqStartInfo.data(), outSubSeqStartInfo.size(), false);
}

void SubNestedSequenceLayer::forward(PassType passType) {
  Layer::forward(passType);

  const Argument& inputSeq = getInput(0);
  CHECK(inputSeq.hasSubseq()) << "The first input of SubNestSequence layer "
                              << "must be a nested sequence.";
  const MatrixPtr selectedIndices = getInputValue(1);
  CHECK_EQ(size_t(inputSeq.getNumSequences()), selectedIndices->getHeight());

  if (dynamic_cast<GpuMatrix*>(selectedIndices.get())) {
    /*
     * Currently, the second input for this layer is generated by
     * kmax_sequence_score_layer whose output is always stored on CPU,
     * or a data_layer which canbe on GPU.
     *
     * If the second input is on GPU, copy it to CPU memory, because this
     * input always uses very few memory, and operations related to it are
     * all logic control, not computations.
     */
    Matrix::resizeOrCreate(selIdsCpu_,
                           selectedIndices->getHeight(),
                           selectedIndices->getWidth(),
                           false /* trans */,
                           false /* useGpu */);
    selIdsCpu_->copyFrom(*selectedIndices);
  } else {
    selIdsCpu_ = selectedIndices;
  }

  Argument::reorganizeSeqInfo(inputSeq.sequenceStartPositions,
                              inputSeq.subSequenceStartPositions,
                              inputSeqInfoVec_);
  calSelectedRows(selIdsCpu_, inputSeqInfoVec_);

  resetOutput(selectedRows_.size(), getSize());
  getOutputValue()->selectRows(*getInputValue(0), *rowIndice_);
}

void SubNestedSequenceLayer::backward(const UpdateCallback& callback) {
  MatrixPtr inputSeqGrad = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();

  if (inputSeqGrad) outputGrad->addToRows(*inputSeqGrad, *rowIndice_);
}

}  // namespace paddle
