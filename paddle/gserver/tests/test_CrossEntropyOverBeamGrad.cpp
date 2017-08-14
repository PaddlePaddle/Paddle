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

struct SingleBeamExpansion {
  vector<int> seqStartPos;
  vector<int> subSeqStartPos;
  vector<real> candidateScores;

  // TODO(caoying): store this into Argument.ids
  vector<real> selectedIndices;
  vector<int> groundTruth;
  vector<int> labelSeqStartPos;
};

void genCandidateScores(bool hasSubSeq,
                        vector<real>& scores,
                        vector<int>& seqStartPos,
                        vector<int>& subSeqStartPos) {}

void genSelectedIndicesAndGroundtruth(size_t beamSize,
                                      vector<int>& seqStartPos,
                                      vector<real>& selectedIndices) {}

SingleBeamExpansion genOneBeam(size_t beamSize, bool hasSubSeq) {
  SingleBeamExpansion beam;
  genCandidateScores(
      hasSubSeq, beam.candidateScores, beam.seqStartPos, beam.subSeqStartPos);
  genSelectedIndicesAndGroundtruth(
      beamSize,
      hasSubSeq ? beam.subSeqStartPos : beam.seqStartPos,
      beam.selectedIndices);
  return beam;
}

void genRandomBeamExpansion(size_t expansionCount,
                            size_t beamSize,
                            vector<SingleBeamExpansion>& beamExpansions) {
  beamExpansions.clear();
  for (size_t i = 0; i < expansionCount; ++i) {
    beamExpansions.emplace_back(genOneBeam(beamSize, i));
  }
}

void testCrossEntropyOverBeam(bool useGpu) {
  TestConfig config;
  config.layerConfig.set_type("cross_entropy_over_beam");

  const size_t expansionCount = 3;
  const size_t beamSize = 3;
  vector<SingleBeamExpansion> beams;
  genRandomBeamExpansion(expansionCount, beamSize, beams);

  size_t seqNum = 0;
  for (size_t i = 0; i < beams.size(); ++i) {
    const SingleBeamExpansion& beam = beams[i];
    // create scores for all the candidates
    MatrixPtr candidateScorePtr =
        Matrix::create(beam.candidateScores.size(), 1, false, false);
    candidateScorePtr->copyFrom(beam.candidateScores.data(),
                                beam.candidateScores.size());

    ostringstream paramName;
    paramName << "candidate_scores_" << i;

    if (beam.subSeqStartPos.size()) {
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
    config.inputDefs.push_back({INPUT_SELF_DEFINE_DATA,
                                paramName.str(),
                                beam.groundTruth,
                                beam.labelSeqStartPos});
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
