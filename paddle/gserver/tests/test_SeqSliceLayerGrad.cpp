/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

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
#include "ModelConfig.pb.h"
#include "paddle/gserver/layers/DataLayer.h"

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

DECLARE_int32(gpu_id);
DECLARE_bool(thread_local_rand_use_global_seed);

const int MAX_SEQ_NUM = 17;
const int MAX_SEQ_LEN = 23;
const int MAX_BEAM_SIZE = 13;

const size_t SEED = (size_t)(time(NULL));

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

void genSeqInfo(vector<int>& seqStartPos, vector<int>& subSeqStartPos) {
  seqStartPos.resize(1, 0);
  subSeqStartPos.resize(1, 0);

  srand(SEED);
  int seqNum = 1 + (rand() % MAX_SEQ_NUM);
  for (int i = 0; i < seqNum; ++i) {
    int subSeqNum = 1 + (rand() % MAX_SEQ_NUM);
    for (int j = 0; j < subSeqNum; ++j)
      subSeqStartPos.push_back(subSeqStartPos.back() +
                               (1 + (rand() % MAX_SEQ_LEN)));
    seqStartPos.push_back(subSeqStartPos.back());
  }
}

/*
  generate start indices according to sequence start positions.
 */
void genStarts(vector<int>& seqStartPos,
               vector<vector<real>>& starts,
               size_t beamSize) {
  starts.clear();
  starts.resize(seqStartPos.size() - 1, vector<real>(beamSize, -1.));

  for (size_t i = 0; i < seqStartPos.size() - 1; ++i) {
    int seqLen = seqStartPos[i + 1] - seqStartPos[i];
    vector<real> randStarts =
        randSampling(seqLen, min(seqLen, static_cast<int>(beamSize)));
    copy(begin(randStarts), end(randStarts), begin(starts[i]));
  }
}

/*
  generate end indices according to sequence start positions and start indices.
 */
void genEnds(vector<int>& seqStartPos,
             vector<vector<real>>& starts,
             vector<vector<real>>& ends,
             size_t beamSize) {
  CHECK_EQ(seqStartPos.size() - 1, starts.size());
  ends.clear();
  ends.resize(seqStartPos.size() - 1, vector<real>(beamSize, -1.));

  for (size_t i = 0; i < starts.size(); ++i) {
    for (size_t j = 0; j < starts[i].size(); ++j) {
      int seqLen = seqStartPos[i + 1] - seqStartPos[i];
      CHECK_GE(seqLen - 1, starts[i][j]);
      if (starts[i][j] == -1.) break;
      if (starts[i][j] == (seqLen - 1)) {
        ends[i][j] = starts[i][j];
      } else {
        ends[i][j] = starts[i][j] + randSampling(seqLen - starts[i][j], 1)[0];
      }
    }
  }
}

void genTestData(vector<int>& seqStartPos,
                 vector<int>& subSeqStartPos,
                 vector<vector<real>>& starts,
                 vector<vector<real>>& ends,
                 bool hasSubseq) {
  size_t beamSize = 1 + (rand() % MAX_BEAM_SIZE);
  genSeqInfo(seqStartPos, subSeqStartPos);

  genStarts(hasSubseq ? subSeqStartPos : seqStartPos, starts, beamSize);
  genEnds(hasSubseq ? subSeqStartPos : seqStartPos, starts, ends, beamSize);
}

template <typename T>
void flatten2dVector(vector<vector<T>>& inVec, vector<T>& outVec) {
  size_t totalSize{0};
  for (auto const& items : inVec) totalSize += items.size();
  outVec.reserve(totalSize);

  for (auto& items : inVec)
    move(items.begin(), items.end(), back_inserter(outVec));
}

void testSeqSliceLayer(bool hasSubseq,
                       bool useGpu,
                       vector<int>& seqStartPos,
                       vector<int>& subSeqStartPos,
                       vector<vector<real>>& starts,
                       vector<vector<real>>& ends) {
  // layer size is not crutial for this layer,
  // so here use a small layer size in the unittest.
  const size_t layerSize{4};
  TestConfig config;
  config.layerConfig.set_type("seq_slice");
  config.layerConfig.set_size(layerSize);

  // add the first input
  MatrixPtr seqInputPtr =
      Matrix::create(hasSubseq ? subSeqStartPos.back() : seqStartPos.back(),
                     layerSize,
                     false,
                     false);
  seqInputPtr->randomizeUniform();

  if (hasSubseq) {
    config.inputDefs.push_back({INPUT_SELF_DEFINE_DATA,
                                "seq_input",
                                seqInputPtr,
                                seqStartPos,
                                subSeqStartPos});
  } else {
    config.inputDefs.push_back(
        {INPUT_SELF_DEFINE_DATA, "seq_input", seqInputPtr, seqStartPos});
  }
  config.layerConfig.add_inputs();

  // add start indices
  if (starts.size()) {
    vector<real> startsToVec;
    flatten2dVector(starts, startsToVec);

    MatrixPtr startMatrixPtr =
        Matrix::create(starts.size(), starts[0].size(), false, false);
    startMatrixPtr->copyFrom(startsToVec.data(), startsToVec.size());

    config.inputDefs.push_back(
        {INPUT_SELF_DEFINE_DATA, "starts", startMatrixPtr});
    config.layerConfig.add_inputs();
    config.layerConfig.set_select_first(true);
  }

  // add end indices
  if (ends.size()) {
    vector<real> endsToVec;
    flatten2dVector(ends, endsToVec);

    MatrixPtr endMatrixPtr =
        Matrix::create(ends.size(), ends[0].size(), false, false);
    endMatrixPtr->copyFrom(endsToVec.data(), endsToVec.size());

    config.inputDefs.push_back({INPUT_SELF_DEFINE_DATA, "ends", endMatrixPtr});
    config.layerConfig.add_inputs();
    config.layerConfig.set_select_first(false);
  }

  testLayerGrad(config, "seq_slice", /*batchSize*/ 100, false, useGpu, false);
}

TEST(Layer, SeqSliceLayer) {
  vector<int> seqStartPos;
  vector<int> subSeqStartPos;
  vector<vector<real>> starts;
  vector<vector<real>> ends;

  std::vector<bool> mode = {false};
#ifdef PADDLE_WITH_CUDA
  mode.push_back(true);
#endif
  genSeqInfo(seqStartPos, subSeqStartPos);
  for (bool hasSubseq : {true, false}) {
    LOG(INFO) << "hasSubSeq : " << hasSubseq;
    genTestData(seqStartPos, subSeqStartPos, starts, ends, hasSubseq);
    for (bool useGpu : mode) {
      vector<vector<real>> tmp;
      testSeqSliceLayer(
          hasSubseq, useGpu, seqStartPos, subSeqStartPos, tmp, ends);
      testSeqSliceLayer(
          hasSubseq, useGpu, seqStartPos, subSeqStartPos, starts, tmp);
      testSeqSliceLayer(
          hasSubseq, useGpu, seqStartPos, subSeqStartPos, starts, ends);
    }
  }
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
