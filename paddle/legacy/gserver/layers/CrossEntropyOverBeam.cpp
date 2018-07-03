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

#include "CrossEntropyOverBeam.h"

namespace paddle {

void CostForOneSequence::calValidExpandStep() {
  validExpansionCount_ = 0;
  goldAsExtraPath_ = true;

  for (size_t i = 0; i < beams_->expansionCount; ++i) {
    real gold = static_cast<real>(beams_->gold[i]);
    if (i) {
      real* start = beams_->candidateIds[i - 1]->getData();
      goldRowIds_[i] = std::count_if(
          start,
          start + goldRowIds_[i - 1] * beamSize_ + goldColIds_[i - 1],
          [](const real& val) { return val != -1.; });
    } else {
      goldRowIds_[i] = 0;
    }

    real* start =
        beams_->candidateIds[i]->getData() + goldRowIds_[i] * beamSize_;
    real* findEnd = std::find(start, start + beamSize_, gold);
    validExpansionCount_++;

    if (start + beamSize_ == findEnd) return;
    goldColIds_[i] = findEnd - start;
  }
  if (goldColIds_[beams_->expansionCount - 1] != -1) goldAsExtraPath_ = false;
}

size_t CostForOneSequence::initLastExpansion() {
  int beamId = validExpansionCount_ - 1;
  const MatrixPtr candidates = beams_->candidateIds[beamId];
  size_t height = candidates->getHeight();

  /* initialization the last expansion. */
  size_t pathCount = std::count_if(candidates->getData(),
                                   candidates->getData() + height * beamSize_,
                                   [](const real& val) { return val != -1; });
  /*
   * if the gold sequence falls off the beam during search, add the gold
   * sequence as the last path into the all expanded candidates.
   */
  if (goldAsExtraPath_) goldIdsInFinalExpansion_ = pathCount++;

  pathRowIdsInEachBeam_.clear();
  pathRowIdsInEachBeam_.resize(validExpansionCount_,
                               std::vector<int>(pathCount, 0));
  parentIdsInBeam_.clear();
  parentIdsInBeam_.resize(pathCount, 0);

  if (goldAsExtraPath_) {
    /* add gold sequence into the total expansion. */
    pathRowIdsInEachBeam_[beamId].back() =
        beams_->gold[beamId] +
        getSeqStartPos(beamId, goldRowIds_[validExpansionCount_ - 1]);
    parentIdsInBeam_.back() = goldRowIds_[validExpansionCount_ - 1];
  } else {
    size_t goldOffset = goldRowIds_[beamId] * beamSize_ + goldColIds_[beamId];
    goldIdsInFinalExpansion_ =
        std::count_if(candidates->getData(),
                      candidates->getData() + goldOffset,
                      [](const real& val) { return val != -1.; });
  }

  /*
   * TODO(caoying): fix this, store the indices of selected candidate
   * paths into Argument.ids
   */
  real* ids = candidates->getData();
  size_t curIdx = 0;
  for (size_t i = 0; i < height; ++i) {
    int basePos = getSeqStartPos(beamId, i);
    for (size_t j = 0; j < beamSize_; ++j) {
      int id = ids[i * beamSize_ + j];
      if (id == -1) continue;
      pathRowIdsInEachBeam_[beamId][curIdx] = id + basePos;
      parentIdsInBeam_[curIdx++] = i;
    }
  }
  return pathCount;
}

void CostForOneSequence::constructTotalExpansion() {
  /*
   * construct the entire expanded beam by begining with the last search
   * in which gold falls off the beam.
   */
  size_t totalPathCount = initLastExpansion();

  for (int beamId = validExpansionCount_ - 2; beamId >= 0; --beamId) {
    const MatrixPtr candidates = beams_->candidateIds[beamId];
    real* ids = candidates->getData();

    int lastParentIdInBeam = -1;
    int basePos = -1;
    for (size_t i = 0;
         i < (goldAsExtraPath_ ? totalPathCount - 1 : totalPathCount);
         ++i) {
      int id = ids[parentIdsInBeam_[i]];
      int parentRowId = std::div(parentIdsInBeam_[i], beamSize_).quot;
      if (parentIdsInBeam_[i] != lastParentIdInBeam)
        basePos = getSeqStartPos(beamId, parentRowId);

      pathRowIdsInEachBeam_[beamId][i] = id + basePos;
      lastParentIdInBeam = parentIdsInBeam_[i];
      parentIdsInBeam_[i] = parentRowId;

      if (goldAsExtraPath_)
        pathRowIdsInEachBeam_[beamId][totalPathCount - 1] =
            beams_->gold[beamId] + getSeqStartPos(beamId, goldRowIds_[beamId]);
    }
  }
}

real CostForOneSequence::globallyNormalizedScore() {
  expandedPathScores_.resize(validExpansionCount_);

  Matrix::resizeOrCreate(
      softmaxOut_, 1, pathRowIdsInEachBeam_[0].size(), false, false);
  softmaxOut_->zeroMem();
  MatrixPtr tmp = Matrix::create(
      softmaxOut_->getData(), softmaxOut_->getWidth(), 1, false, false);

  for (size_t i = 0; i < validExpansionCount_; ++i) {
    Matrix::resizeOrCreate(expandedPathScores_[i],
                           pathRowIdsInEachBeam_[i].size(),
                           1,
                           false,
                           false);
    expandedPathScores_[i]->zeroMem();

    IVectorPtr rowIds = IVector::create(pathRowIdsInEachBeam_[i].data(),
                                        pathRowIdsInEachBeam_[i].size(),
                                        false);
    expandedPathScores_[i]->selectRows(*(beams_->scores[i]), *rowIds);
    tmp->add(*expandedPathScores_[i]);
  }

  softmaxOut_->softmax(*softmaxOut_);
  return -std::log(softmaxOut_->getData()[goldIdsInFinalExpansion_]);
}

real CostForOneSequence::forward() {
  calValidExpandStep();
  constructTotalExpansion();
  return globallyNormalizedScore();
}

void CostForOneSequence::backward() {
  /*
   * when softmax layer is the output layer, and it is combined with
   * cross-entropy as cost. The derivate with regard to softmax's input
   * is simply:
   *
   * grad_i = softmax_out_i - target_i,
   *
   * and here hard label is used.
   */
  softmaxOut_->getData()[goldIdsInFinalExpansion_] -= 1.;

  MatrixPtr tmp = Matrix::create(
      softmaxOut_->getData(), softmaxOut_->getWidth(), 1, false, false);

  for (size_t i = 0; i < validExpansionCount_; ++i) {
    IVectorPtr rowIds = IVector::create(pathRowIdsInEachBeam_[i].data(),
                                        pathRowIdsInEachBeam_[i].size(),
                                        false);
    /*
      beams_->scoreGrad[i] has been intialized outside this class, this
      class only keeps a pointer pointing to the original input gradients,
      so here does not need to allocate or initalize the memory.
    */
    tmp->addToRows(*beams_->scoreGrad[i], *rowIds);
  }
}

REGISTER_LAYER(cross_entropy_over_beam, CrossEntropyOverBeam);

bool CrossEntropyOverBeam::init(const LayerMap& layerMap,
                                const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(0U, inputLayers_.size() % 3) << "Error input number.";

  beamExpanCount_ = inputLayers_.size() / 3;

  candidateScores_.resize(beamExpanCount_);
  candidateScoreGrad_.resize(beamExpanCount_);

  candidateInBeam_.resize(beamExpanCount_);
  goldSequence_.resize(beamExpanCount_);
  gradToInputs_.resize(beamExpanCount_);

  setNeedSequenceInfo(false);
  return true;
}

void CrossEntropyOverBeam::checkInputs() {
  batchSize_ = 0;
  for (size_t i = 0; i < beamExpanCount_; ++i) {
    const Argument& scores = getInput(i * 3);
    const Argument& selCandidates = getInput(i * 3 + 1);
    const Argument& goldSeq = getInput(i * 3 + 2);

    if (i) {
      CHECK(scores.hasSubseq()) << "input " << i << " "
                                << inputLayers_[i * 3]->getName()
                                << " should be a nested sequence";
      CHECK_EQ(getInputValue(i * 3 + 1)->getWidth(), beamSize_);
      CHECK_EQ(batchSize_, static_cast<size_t>(scores.getNumSequences()));
      CHECK_EQ(scores.getNumSubSequences(), selCandidates.getBatchSize());
    } else {
      CHECK(scores.hasSeq()) << "input " << i << " "
                             << inputLayers_[i]->getName()
                             << " should be a sequence";
      batchSize_ = scores.getNumSequences();
      beamSize_ = getInputValue(i * 3 + 1)->getWidth();
      CHECK_EQ(batchSize_, static_cast<size_t>(selCandidates.getBatchSize()));
    }
    CHECK_EQ(1U, scores.value->getWidth());
    CHECK_EQ(batchSize_, static_cast<size_t>(goldSeq.getBatchSize()));
  }
}

void CrossEntropyOverBeam::copyInputsToCpu() {
  auto copyValue = [](const MatrixPtr& src, MatrixPtr& trg) {
    if (dynamic_cast<GpuMatrix*>(src.get())) {
      Matrix::resizeOrCreate(
          trg, src->getHeight(), src->getWidth(), false, false);
      trg->copyFrom(*src);
    } else {
      trg = std::move(src);
    }
  };

  auto copyIds = [](const IVectorPtr& src, IVectorPtr& trg) {
    if (dynamic_cast<GpuIVector*>(src.get())) {
      IVector::resizeOrCreate(trg, src->getSize(), false);
      trg->copyFrom(*src);
    } else {
      trg = std::move(src);
    }
  };

  beamSplitPos_.clear();
  beamSplitPos_.resize(batchSize_, std::vector<int>(beamExpanCount_, 0));
  for (size_t i = 0; i < beamExpanCount_; ++i) {
    copyValue(getInputValue(i * 3), candidateScores_[i]);
    copyValue(getInputValue(i * 3 + 1), candidateInBeam_[i]);
    copyIds(getInput(i * 3 + 2).ids, goldSequence_[i]);

    if (i) {
      ICpuGpuVectorPtr seqInfo = getInput(i * 3).sequenceStartPositions;
      const int* seqStarts = seqInfo->getMutableData(false);
      ICpuGpuVectorPtr subSeqInfo = getInput(i * 3).subSequenceStartPositions;
      const int* subSeqStarts = subSeqInfo->getMutableData(false);

      size_t seqId = 1;
      for (size_t subSeqId = 0; subSeqId < subSeqInfo->getSize() - 1;
           ++subSeqId) {
        CHECK_LT(seqId, seqInfo->getSize());
        if (subSeqStarts[subSeqId] == seqStarts[seqId]) {
          beamSplitPos_[seqId][i] = beamSplitPos_[seqId - 1][i];
          seqId++;
        }
        beamSplitPos_[seqId - 1][i]++;
      }
    } else {
      for (size_t j = 0; j < batchSize_; ++j) beamSplitPos_[j][i] = j + 1;
    }
  }
}

void CrossEntropyOverBeam::splitBatchBeams() {
  beamCosts_.resize(batchSize_);
  beamPerSeq_.resize(batchSize_, BeamExpansion(beamExpanCount_));

  for (size_t i = 0; i < beamExpanCount_; ++i) {
    int* seqStarts =
        getInput(i * 3).sequenceStartPositions->getMutableData(false);

    int* subSeqStarts = nullptr;
    int maxLen = 0;
    if (i) {
      subSeqStarts =
          getInput(i * 3).subSequenceStartPositions->getMutableData(false);
      maxLen = getInput(i * 3).subSequenceStartPositions->getSize() - 1;
    } else {
      maxLen = getInput(i).sequenceStartPositions->getSize() - 1;
    }

    for (size_t j = 0; j < batchSize_; ++j) {
      beamPerSeq_[j].scores[i] =
          Matrix::create(candidateScores_[i]->getData() + seqStarts[j],
                         seqStarts[j + 1] - seqStarts[j],
                         1,
                         false,
                         false);
      beamPerSeq_[j].scoreGrad[i] =
          Matrix::create(candidateScoreGrad_[i]->getData() + seqStarts[j],
                         seqStarts[j + 1] - seqStarts[j],
                         1,
                         false,
                         false);

      int offset = j ? beamSplitPos_[j - 1][i] : 0;
      int height = beamSplitPos_[j][i] - (j ? beamSplitPos_[j - 1][i] : 0);
      CHECK_GE(maxLen, offset + height);
      beamPerSeq_[j].seqInfo[i] = IVector::create(
          (i ? subSeqStarts : seqStarts) + offset, height + 1, false);

      beamPerSeq_[j].candidateIds[i] =
          Matrix::create(candidateInBeam_[i]->getData() + offset * beamSize_,
                         height,
                         beamSize_,
                         false,
                         false);
      beamPerSeq_[j].gold[i] = goldSequence_[i]->getData()[j];

      CHECK_LE(beamPerSeq_[j].gold[i], seqStarts[j + 1] - seqStarts[j]);
    }
  }
}

void CrossEntropyOverBeam::resizeOutput() {
  Matrix::resizeOrCreate(output_.value, batchSize_, 1, false, false);
  output_.value->zeroMem();

  for (size_t i = 0; i < beamExpanCount_; ++i) {
    MatrixPtr inGrad = getInputGrad(i * 3);
    if (dynamic_cast<GpuMatrix*>(inGrad.get())) {
      Matrix::resizeOrCreate(candidateScoreGrad_[i],
                             inGrad->getHeight(),
                             inGrad->getWidth(),
                             false,
                             false);
    } else {
      candidateScoreGrad_[i] = std::move(inGrad);
    }
    candidateScoreGrad_[i]->zeroMem();
  }
}

void CrossEntropyOverBeam::copyGradToGpu(size_t copyCount) {
  for (size_t i = 0; i < beamExpanCount_; ++i) {
    if (dynamic_cast<GpuMatrix*>(getInputGrad(i * 3).get()))
      getInputGrad(i * 3)->copyFrom(*candidateScoreGrad_[i]);

    if (i == copyCount - 1) break;
  }
}

void CrossEntropyOverBeam::forward(PassType passType) {
  Layer::forward(passType);

  checkInputs();
  copyInputsToCpu();

  resizeOutput();
  splitBatchBeams();

  MatrixPtr outputValue = getOutputValue();
  for (size_t i = 0; i < batchSize_; ++i) {
    BeamExpansionPtr ptr = std::make_shared<BeamExpansion>(beamPerSeq_[i]);
    beamCosts_[i].setData(std::move(ptr), beamSize_);
    outputValue->getData()[i] = beamCosts_[i].forward();
  }
}

void CrossEntropyOverBeam::backward(const UpdateCallback& callback) {
  for (size_t i = 0; i < batchSize_; ++i) {
    beamCosts_[i].backward();
    copyGradToGpu(beamCosts_[i].getValidExpansionCount());
  }
}

}  // namespace paddle
