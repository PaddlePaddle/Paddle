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

#pragma once

#include "CrossEntropyOverBeam.h"
#include "Layer.h"

namespace paddle {

/* This struct stores the beams in all search steps for a single sequence. */
struct BeamExpansion {
  std::vector<MatrixPtr> scores;
  std::vector<IVectorPtr> seqInfo;

  std::vector<MatrixPtr> candidateIds;
  std::vector<int> gold;

  std::vector<MatrixPtr> scoreGrad;

  size_t expansionCount;

  explicit BeamExpansion(int n) {
    expansionCount = n;
    scores.resize(expansionCount);
    seqInfo.resize(expansionCount);
    candidateIds.resize(expansionCount);
    scoreGrad.resize(expansionCount);

    gold.resize(expansionCount);
  }
};
typedef std::shared_ptr<BeamExpansion> BeamExpansionPtr;

class CostForOneSequence {
 public:
  CostForOneSequence()
      : beamSize_(0), validExpansionCount_(0), goldAsExtraPath_(false) {}
  void setData(const BeamExpansionPtr bPtr, size_t beamSize) {
    beams_ = bPtr;
    beamSize_ = beamSize;

    expandedPathScores_.clear();
    expandedPathScores_.resize(beams_->expansionCount);

    goldRowIds_.clear();
    goldRowIds_.resize(beams_->expansionCount, 0);
    goldColIds_.clear();
    goldColIds_.resize(beams_->expansionCount, -1);
  }
  size_t getValidExpansionCount() { return validExpansionCount_; }

  real forward();
  void backward();

 private:
  void calValidExpandStep();
  void constructTotalExpansion();
  size_t initLastExpansion();
  real globallyNormalizedScore();

  int getSeqStartPos(size_t beamId, size_t rowId) {
    CHECK_GT(beams_->seqInfo[beamId]->getSize() - 1, rowId);
    int* starts = beams_->seqInfo[beamId]->getData();
    return starts[rowId] - starts[0];
  }

  size_t beamSize_;
  size_t validExpansionCount_;
  bool goldAsExtraPath_;
  std::vector<int> goldRowIds_;
  std::vector<int> goldColIds_;

  BeamExpansionPtr beams_;
  std::vector<std::vector<int>> pathRowIdsInEachBeam_;
  std::vector<int> parentIdsInBeam_;
  size_t goldIdsInFinalExpansion_;

  std::vector<MatrixPtr> expandedPathScores_;

  MatrixPtr softmaxOut_;
};

class CrossEntropyOverBeam : public Layer {
 public:
  explicit CrossEntropyOverBeam(const LayerConfig& config) : Layer(config) {}
  bool init(const LayerMap& layerMap,
            const ParameterMap& parameterMap) override;
  void forward(PassType passType) override;
  void backward(const UpdateCallback& callback) override;

 private:
  void checkInputs();
  void copyInputsToCpu();
  void resizeOutput();
  void copyGradToGpu(size_t copyCount);
  void splitBatchBeams();

  size_t beamExpanCount_;
  size_t batchSize_;
  size_t beamSize_;

  /*
   * the process of constructing beams is not friendly to GPU, currently, this
   * layer only runs on CPU, if any of its inputs is on GPU memory, then copy
   * it to CPU memory.
   */
  std::vector<MatrixPtr> candidateScores_;
  std::vector<MatrixPtr> candidateScoreGrad_;
  std::vector<MatrixPtr> candidateInBeam_;
  std::vector<MatrixPtr> gradToInputs_;
  std::vector<IVectorPtr> goldSequence_;
  std::vector<std::vector<int>> beamSplitPos_;

  /*
   * split entire bath of beams into beam per sequnence and store the result
   * into this member.
   */
  std::vector<BeamExpansion> beamPerSeq_;
  /* beamCosts_ is used to propagate error in one sequence. */
  std::vector<CostForOneSequence> beamCosts_;
};

}  // namespace paddle
