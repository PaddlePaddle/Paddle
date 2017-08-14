/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <random>
#include <sstream>

#include <gtest/gtest.h>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/trainer/Trainer.h"

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT

DECLARE_int32(gpu_id);
DECLARE_bool(thread_local_rand_use_global_seed);

const size_t MAX_SEQ_NUM = 10;
const size_t MAX_SEQ_LEN = 27;
const size_t MAX_BEAM_SIZE = 10;

struct SingleBeamExpansion {
  vector<int> seqStartPos;
  vector<int> subSeqStartPos;
  vector<real> candidateScores;

  // TODO(caoying): store this into Argument.ids
  vector<real> selectedIndices;

  vector<int> groundTruth;
  vector<size_t> inBeam;
  vector<int> rowIdxInBeam;
};

void genRand(real* numbers, size_t n) {
  default_random_engine generator;
  uniform_real_distribution<double> distribution(0.0, 1.0);
  for (size_t i = 0; i < n; ++i) numbers[i] = distribution(generator);
}

vector<real> randSampling(real range, int n) {
  CHECK_GE(range, n);
  vector<real> num(range);
  iota(begin(num), end(num), 0.);
  if (range == n) return num;

  random_shuffle(begin(num), end(num));
  num.resize(n);
  sort(begin(num), end(num));
  return num;
}

void genCandidateScores(bool hasSubseq,
                        size_t beamSize,
                        SingleBeamExpansion& prevBeam,
                        SingleBeamExpansion& curBeam) {
  vector<int>& seqStartPos = curBeam.seqStartPos;
  seqStartPos.resize(1, 0);
  vector<int>& subSeqStartPos = curBeam.subSeqStartPos;
  subSeqStartPos.resize(1, 0);

  srand((size_t)(time(NULL)));
  // srand(1);
  if (prevBeam.selectedIndices.size()) {
    if (prevBeam.subSeqStartPos.size() > 1) {
      int seqIdx = 1;
      // samples in previous beam are nested sequences.
      for (size_t i = 1; i < prevBeam.subSeqStartPos.size(); ++i) {
        for (size_t j = 0; j < beamSize; ++j) {
          if (prevBeam.selectedIndices[(i - 1) * beamSize + j] == -1.) break;
          for (size_t k = 0; k < beamSize; ++k)
            subSeqStartPos.push_back(1 + (rand() % MAX_SEQ_LEN) +
                                     subSeqStartPos.back());
        }
        if (prevBeam.seqStartPos[seqIdx] == prevBeam.subSeqStartPos[i]) {
          seqStartPos.push_back(subSeqStartPos.back());
          seqIdx++;
        }
      }
    } else {
      // samples in previous beam are sequences.
      for (size_t i = 0; i <= prevBeam.selectedIndices.size(); ++i) {
        if (i && i % beamSize == 0) {
          seqStartPos.push_back(subSeqStartPos.back());
          if (i == prevBeam.selectedIndices.size()) break;
        }
        if (prevBeam.selectedIndices[i] == -1.) continue;
        subSeqStartPos.push_back(subSeqStartPos.back() +
                                 (1 + (rand() % MAX_SEQ_LEN)));
      }
    }
  } else {
    // the first beam expansion
    int seqNum = 1 + (rand() % MAX_SEQ_NUM);
    for (int i = 0; i < seqNum; ++i) {
      if (hasSubseq) {
        for (size_t j = 0; j < 1 + (rand() % MAX_SEQ_NUM); ++j)
          subSeqStartPos.push_back(subSeqStartPos.back() +
                                   (1 + (rand() % MAX_SEQ_LEN)));
        seqStartPos.push_back(subSeqStartPos.back());
      } else {
        seqStartPos.push_back(seqStartPos.back() +
                              (1 + (rand() % MAX_SEQ_LEN)));
      }
    }
  }

  size_t totalSeqNum = hasSubseq ? subSeqStartPos.back() : seqStartPos.back();
  curBeam.candidateScores.resize(totalSeqNum, 0.);
  genRand(curBeam.candidateScores.data(), totalSeqNum);
}

void genSelectedIndices(size_t beamSize,
                        vector<int>& seqStartPos,
                        vector<real>& selectedIndices) {
  size_t selectedIdsCount = beamSize * (seqStartPos.size() - 1);
  selectedIndices.resize(selectedIdsCount, -1.);

  for (size_t i = 0; i < seqStartPos.size() - 1; ++i) {
    int seqLen = seqStartPos[i + 1] - seqStartPos[i];
    int n = min(seqLen, static_cast<int>(beamSize));
    vector<real> ids = randSampling(seqLen, n);
    memcpy(selectedIndices.data() + i * beamSize,
           ids.data(),
           sizeof(real) * ids.size());
  }
}

void genGroundTruth(vector<SingleBeamExpansion>& beamExpansions,
                    size_t beamSize) {
  size_t seqNum = beamExpansions[1].seqStartPos.size() - 1;
  for (size_t i = 2; i < beamExpansions.size(); ++i)
    CHECK_EQ(seqNum, beamExpansions[i - 1].seqStartPos.size() - 1);

  // srand(1);
  srand((size_t)(time(NULL)));

  // initialize the first beam.
  SingleBeamExpansion& beam = beamExpansions[1];
  beam.groundTruth.resize(seqNum, 0);
  beam.inBeam.resize(seqNum, 0);
  beam.rowIdxInBeam.resize(seqNum, -1);

  auto begPos = beam.selectedIndices.begin();
  for (size_t i = 0; i < seqNum; ++i) {
    int seqLen = beam.seqStartPos[i + 1] - beam.seqStartPos[i];
    int label = rand() % seqLen;
    auto endPos = begPos + beamSize;
    beam.groundTruth[i] = label;
    if (find(begPos, endPos, real(label)) != endPos) beam.inBeam[i] = 1;
    begPos = endPos;
    beam.rowIdxInBeam[i] = i;
  }

  // iterate over each beam expansions
  for (size_t i = 2; i < beamExpansions.size(); ++i) {
    SingleBeamExpansion& curBeam = beamExpansions[i];
    SingleBeamExpansion& prevBeam = beamExpansions[i - 1];

    curBeam.groundTruth.resize(seqNum, 0);
    curBeam.inBeam.resize(seqNum, 0);
    curBeam.rowIdxInBeam.resize(seqNum, -1);

    // iterate over each sequence
    for (size_t j = 0; j < seqNum; ++j) {
      if (prevBeam.inBeam[j]) {
        // gold sequence falls in the beam in previous search.

        auto begPos = prevBeam.selectedIndices.begin();
        auto endPos = begPos + prevBeam.rowIdxInBeam[j] * beamSize;
        size_t totalExpansion =
            prevBeam.rowIdxInBeam[j] * beamSize - count(begPos, endPos, -1.);
        curBeam.rowIdxInBeam[j] = totalExpansion + prevBeam.groundTruth[j];

        CHECK_LE(curBeam.rowIdxInBeam[j] + 1,
                 curBeam.subSeqStartPos.size() - 1);
        int start = curBeam.subSeqStartPos[curBeam.rowIdxInBeam[j]];
        int end = curBeam.subSeqStartPos[curBeam.rowIdxInBeam[j] + 1];
        CHECK_GT(size_t(end), size_t(start));
        int label = rand() % (end - start);

        curBeam.groundTruth[j] = label;
        auto findBeg = curBeam.selectedIndices.begin() +
                       curBeam.rowIdxInBeam[j] * beamSize;
        auto findEnd = findBeg + beamSize;
        if (find(findBeg, findEnd, real(label)) != findEnd)
          curBeam.inBeam[j] = 1;
      } else {
        // in previous search, gold sequence has fallen off the beam,
        // the beam search stops, here use -1 as a dummy label.
        // It will not used in calculation the cost.
        beamExpansions[i].groundTruth[j] = -1;
      }
    }
  }
}

void genOneBeam(size_t beamSize,
                bool hasSubseq,
                SingleBeamExpansion& prevBeam,
                SingleBeamExpansion& curBeam) {
  genCandidateScores(hasSubseq, beamSize, prevBeam, curBeam);
  genSelectedIndices(beamSize,
                     hasSubseq ? curBeam.subSeqStartPos : curBeam.seqStartPos,
                     curBeam.selectedIndices);
}

void genRandomBeamExpansion(size_t expansionCount,
                            size_t beamSize,
                            vector<SingleBeamExpansion>& beamExpansions) {
  beamExpansions.clear();
  beamExpansions.resize(expansionCount + 1);

  // beamExpansions[0] is reserved.
  for (size_t i = 1; i <= expansionCount; ++i)
    genOneBeam(beamSize, bool(i - 1), beamExpansions[i - 1], beamExpansions[i]);
  genGroundTruth(beamExpansions, beamSize);
}

void testCrossEntropyOverBeam(bool useGpu) {
  TestConfig config;
  config.layerConfig.set_type("cross_entropy_over_beam");

  const size_t expansionCount = 3;
  const size_t beamSize = MAX_BEAM_SIZE;
  vector<SingleBeamExpansion> beams;
  genRandomBeamExpansion(expansionCount, beamSize, beams);

  size_t seqNum = 0;
  for (size_t i = 1; i < beams.size(); ++i) {
    const SingleBeamExpansion& beam = beams[i];
    // create scores for all the candidates
    MatrixPtr candidateScorePtr =
        Matrix::create(beam.candidateScores.size(), 1, false, false);
    candidateScorePtr->copyFrom(beam.candidateScores.data(),
                                beam.candidateScores.size());

    ostringstream paramName;
    paramName << "candidate_scores_" << i;

    if (beam.subSeqStartPos.size() > 1) {
      seqNum = beam.subSeqStartPos.size() - 1;
      config.inputDefs.push_back({INPUT_SELF_DEFINE_DATA,
                                  paramName.str(),
                                  candidateScorePtr,
                                  beam.seqStartPos,
                                  beam.subSeqStartPos});
    } else {
      seqNum = beam.seqStartPos.size() - 1;
      config.inputDefs.push_back({INPUT_SELF_DEFINE_DATA,
                                  paramName.str(),
                                  candidateScorePtr,
                                  beam.seqStartPos});
    }
    config.layerConfig.add_inputs();

    // create indices for the selected candidates
    MatrixPtr selectedCandidates =
        Matrix::create(seqNum, beamSize, false, false);
    selectedCandidates->copyFrom(beam.selectedIndices.data(),
                                 beam.selectedIndices.size());
    paramName.clear();
    paramName << "selected_candidates_" << i;
    config.inputDefs.push_back(
        {INPUT_SELF_DEFINE_DATA, paramName.str(), selectedCandidates});
    config.layerConfig.add_inputs();

    // create the ground truth
    paramName.clear();
    paramName << "label_" << i;
    config.inputDefs.push_back(
        {INPUT_SELF_DEFINE_DATA, paramName.str(), beam.groundTruth});
    config.layerConfig.add_inputs();
  }

  testLayerGrad(
      config, "cross_entropy_over_beam", seqNum, false, useGpu, false);
}

TEST(Layer, CrossEntropyOverBeam) {
  for (bool useGpu : {false, true}) testCrossEntropyOverBeam(useGpu);
}

int main(int argc, char** argv) {
  initMain(argc, argv);
  hl_start();
  hl_init(FLAGS_gpu_id);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
