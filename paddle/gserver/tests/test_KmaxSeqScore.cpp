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

#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/trainer/Trainer.h"
#include "paddle/utils/GlobalConstants.h"

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_bool(use_gpu);
DECLARE_int32(gpu_id);
DECLARE_bool(thread_local_rand_use_global_seed);

vector<int> randSampling(int range, int n) {
  srand(1);
  CHECK_GE(range, n);
  vector<int> num(range);
  iota(begin(num), end(num), 0);
  if (range == n) return num;

  random_shuffle(begin(num), end(num));
  num.resize(n);
  return num;
}

void genRandomSeqInfo(vector<int>& seqStartPosition,
                      vector<int>& subSeqStartPosition) {
  const int maxSeqNum = 5;
  // generate random start position information
  int seqNum = 1 + (rand() % maxSeqNum);
  seqStartPosition.resize(seqNum + 1, 0);
  subSeqStartPosition.resize(1, 0);

  for (int i = 0; i < seqNum; ++i) {
    int subSeqLen = 1 + (rand() % maxSeqNum);
    for (int j = 0; j < subSeqLen; ++j)
      subSeqStartPosition.push_back(subSeqStartPosition.back() + subSeqLen);
    seqStartPosition[i + 1] = subSeqStartPosition.back();
  }
}

void genRandomGroundTruth(real* values,
                          vector<vector<int>>& groundTruth,
                          vector<int>& seqStartPosition,
                          vector<int>& subSeqStartPosition,
                          bool useSubseqInfo,
                          size_t beamSize) {
  auto genData = [&](real* values, vector<int>& startPos, size_t beamSize) {
    groundTruth.resize(startPos.size() - 1, vector<int>(beamSize, -1));

    for (size_t i = 0; i < startPos.size() - 1; ++i) {
      int seqLen = startPos[i + 1] - startPos[i];
      vector<int> pos =
          randSampling(seqLen, min(static_cast<int>(beamSize), seqLen));
      for (size_t j = 0; j < pos.size(); ++j) {
        groundTruth[i][j] = pos[j];
        values[subSeqStartPosition[i] + pos[j]] = 1.;
      }
    }
  };

  if (useSubseqInfo)
    genData(values, subSeqStartPosition, beamSize);
  else
    genData(values, seqStartPosition, beamSize);
}

// Test that the batchNormLayer can be followed by a ConvLayer
TEST(Layer, kmaxSeqScoreLayer) {
  const size_t beamSize = 5;

  vector<int> seqStartPosition;
  vector<int> subSeqStartPosition;
  genRandomSeqInfo(seqStartPosition, subSeqStartPosition);
  MatrixPtr inValue =
      Matrix::create(subSeqStartPosition.back(), 1, false, false);
  inValue->randomizeUniform();

  for (auto hasSubseq : {false, true}) {
    vector<vector<int>> groundTruth;
    genRandomGroundTruth(inValue->getData(),
                         groundTruth,
                         seqStartPosition,
                         subSeqStartPosition,
                         hasSubseq,
                         beamSize);

    for (auto useGpu : {false, true}) {
      TestConfig config;
      config.layerConfig.set_type("kmax_seq_score");
      config.layerConfig.set_beam_size(beamSize);
      config.inputDefs.push_back(
          {hasSubseq ? INPUT_HASSUB_SEQUENCE_DATA : INPUT_SEQUENCE_DATA,
           "layer_0",
           1,
           0});
      config.layerConfig.add_inputs();

      // data layer initialize
      std::vector<DataLayerPtr> dataLayers;
      LayerMap layerMap;
      vector<Argument> datas;
      initDataLayer(config,
                    &dataLayers,
                    &datas,
                    &layerMap,
                    "kmax_seq_score",
                    100,
                    false,
                    useGpu);
      // test layer initialize
      std::vector<ParameterPtr> parameters;
      LayerPtr kmaxSeqScoreLayer;
      initTestLayer(config, &layerMap, &parameters, &kmaxSeqScoreLayer);
      kmaxSeqScoreLayer->forward(PASS_TRAIN);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand(1);
  return RUN_ALL_TESTS();
}
