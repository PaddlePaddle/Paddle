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

#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <vector>
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"
#include "paddle/utils/GlobalConstants.h"

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_bool(use_gpu);
DECLARE_int32(gpu_id);
DECLARE_bool(thread_local_rand_use_global_seed);

vector<int> randSampling(int range, int n) {
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
  const int maxSeqNum = 100;
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
                          vector<int>& startPos,
                          size_t beamSize) {
  groundTruth.resize(startPos.size() - 1, vector<int>(beamSize, -1));
  for (size_t i = 0; i < startPos.size() - 1; ++i) {
    int seqLen = startPos[i + 1] - startPos[i];
    vector<int> pos =
        randSampling(seqLen, min(static_cast<int>(beamSize), seqLen));
    for (size_t j = 0; j < pos.size(); ++j) {
      groundTruth[i][j] = pos[j];
      values[startPos[i] + pos[j]] = 1.;
    }
  }
}

void checkLayerOut(vector<vector<int>> groundTruth,
                   real* layerOut,
                   size_t beamSize) {
  for (size_t i = 0; i < groundTruth.size(); ++i) {
    int begPos = i * beamSize;
    vector<real> tmp(layerOut + begPos, layerOut + begPos + beamSize);
    sort(begin(tmp), end(tmp));
    sort(begin(groundTruth[i]), end(groundTruth[i]));
    for (size_t j = 0; j < beamSize; ++j) CHECK_EQ(tmp[j], groundTruth[i][j]);
  }
}

TEST(Layer, kmaxSeqScoreLayer) {
  const size_t maxBeamSize = 100;
  size_t beamSize = 1 + (rand() % maxBeamSize);

  vector<int> seqStartPosition;
  vector<int> subSeqStartPosition;
  genRandomSeqInfo(seqStartPosition, subSeqStartPosition);
  MatrixPtr inValue =
      Matrix::create(subSeqStartPosition.back(), 1, false, false);

  std::vector<bool> mode = {false};
#ifdef PADDLE_WITH_CUDA
  mode.push_back(true);
#endif

  for (auto hasSubseq : {false, true}) {
    vector<vector<int>> groundTruth;
    inValue->randomizeUniform();
    genRandomGroundTruth(inValue->getData(),
                         groundTruth,
                         hasSubseq ? subSeqStartPosition : seqStartPosition,
                         beamSize);

    for (auto useGpu : mode) {
      TestConfig config;
      config.layerConfig.set_type("kmax_seq_score");
      config.layerConfig.set_beam_size(beamSize);

      if (hasSubseq) {
        config.inputDefs.push_back({INPUT_SELF_DEFINE_DATA,
                                    "scores",
                                    inValue,
                                    seqStartPosition,
                                    subSeqStartPosition});
      } else {
        config.inputDefs.push_back(
            {INPUT_SELF_DEFINE_DATA, "scores", inValue, seqStartPosition});
      }
      config.layerConfig.add_inputs();

      // data layer initialize
      std::vector<DataLayerPtr> dataLayers;
      LayerMap layerMap;
      vector<Argument> datas;
      initDataLayer(
          config,
          &dataLayers,
          &datas,
          &layerMap,
          "kmax_seq_score",
          100 /* actually this parameter is unused in self-defined input*/,
          false,
          useGpu);
      // test layer initialize
      std::vector<ParameterPtr> parameters;
      LayerPtr kmaxSeqScoreLayer;
      FLAGS_use_gpu = useGpu;
      initTestLayer(config, &layerMap, &parameters, &kmaxSeqScoreLayer);
      kmaxSeqScoreLayer->forward(PASS_TRAIN);

      const MatrixPtr outValue = kmaxSeqScoreLayer->getOutputValue();
      CHECK_EQ(outValue->getHeight(),
               hasSubseq ? subSeqStartPosition.size() - 1
                         : seqStartPosition.size() - 1);
      CHECK_EQ(outValue->getWidth(), beamSize);
      checkLayerOut(groundTruth, outValue->getData(), beamSize);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  FLAGS_thread_local_rand_use_global_seed = true;
  srand((size_t)(time(NULL)));
  return RUN_ALL_TESTS();
}
