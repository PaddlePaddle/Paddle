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
#include "paddle/gserver/layers/LinearChainCRF.h"

#include "LayerGradUtil.h"
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT

DECLARE_int32(gpu_id);
DECLARE_bool(thread_local_rand_use_global_seed);

static inline bool getNextSequence(std::vector<int>& seq, int numClasses) {
  for (auto& v : seq) {
    if (++v < numClasses) {
      return true;
    }
    v = 0;
  }
  return false;
}

// log(exp(x) + exp(y))
static inline real logSum(real x, real y) {
  real maxValue = std::max(x, y);
  if (std::isinf(maxValue)) {
    return -std::numeric_limits<real>::infinity();
  } else {
    return maxValue + log(exp(x - maxValue) + exp(y - maxValue));
  }
}

static inline std::vector<int> genRandLabels(int numClasses, int length) {
  std::vector<int> labels(length);
  for (int i = 0; i < length; ++i) {
    labels[i] = rand() % numClasses;  // NOLINT
  }
  return labels;
}

TEST(CRFLayer, cost) {
  const int numClasses = 4;
  CpuVector para(numClasses * (numClasses + 2));
  real* a = para.getData();
  real* b = para.getData() + numClasses;
  real* w = para.getData() + 2 * numClasses;
  LinearChainCRF crf(4, para.getData());
  for (int length : {1, 2, 3, 10}) {
    for (int tries = 0; tries < 10; ++tries) {
      CpuMatrix x(length, numClasses);
      x.randomizeUniform();
      para.randnorm(0, 2);

      std::vector<int> goldenLabels = genRandLabels(numClasses, length);

      real cost = crf.forward(x.getData(), goldenLabels.data(), length);

      real logZ = -std::numeric_limits<real>::infinity();
      real logNominator = -std::numeric_limits<real>::infinity();
      std::vector<int> testResult(length, 0);
      do {
        real score = a[testResult.front()];
        score += x.getElement(0, testResult.front());
        for (int k = 1; k < length; ++k) {
          score += x.getElement(k, testResult[k]) +
                   w[numClasses * testResult[k - 1] + testResult[k]];
        }
        score += b[testResult.back()];
        logZ = logSum(logZ, score);

        if (goldenLabels == testResult) {
          logNominator = score;
        }
      } while (getNextSequence(testResult, numClasses));

      real trueCost = -logNominator + logZ;

      real diff = fabs(trueCost - cost);
      diff /= fabs(cost) < fabs(trueCost) ? fabs(cost) : fabs(trueCost);
      VLOG(1) << "cost=" << cost << " trueCost=" << trueCost << " diff=" << diff
              << std::endl;
      if (typeid(real) == typeid(double)) {  // NOLINT
        EXPECT_LE(diff, 1e-10);
      } else {
        EXPECT_LE(diff, 5e-3);
      }
    }
  }
}

inline real epsilon() { return typeid(real) == typeid(double) ? 1e-10 : 0.06; }

TestConfig initTestConfig(size_t numClasses, bool withWeight) {
  TestConfig config;
  config.layerConfig.set_type("crf");
  config.layerConfig.set_size(numClasses);
  config.biasSize = 0;

  config.inputDefs.push_back({INPUT_SEQUENCE_DATA,
                              "layer_0",
                              numClasses,
                              numClasses * (numClasses + 2)});
  config.layerConfig.add_inputs();
  config.inputDefs.push_back(
      {INPUT_SEQUENCE_LABEL, "layer_label", numClasses, 0});
  config.layerConfig.add_inputs();

  if (withWeight) {
    config.inputDefs.push_back({INPUT_DENSE_DIM_DATA, "layer_weight", 1, 0});
    config.layerConfig.add_inputs();
  }

  return config;
}

TEST(Layer, CRFLayer) {
  size_t numClasses = 10;
  for (int tries = 0; tries < 5; ++tries) {
    TestConfig config = initTestConfig(numClasses, /* withWeight= */ false);
    for (int length : {1, 3, 100}) {
      // Not support GPU now
      testLayerGrad(config,
                    "crf",
                    length,
                    /* trans= */ false,
                    /* useGpu= */ false,
                    /* useWeight= */ false,
                    epsilon());
    }
  }
}

TEST(Layer, CRFLayerUseWeight) {
  size_t numClasses = 10;
  for (int tries = 0; tries < 5; ++tries) {
    TestConfig config = initTestConfig(numClasses, /* withWeight= */ true);
    for (int length : {1, 3, 100}) {
      // Not support GPU now
      testLayerGrad(config,
                    "crf",
                    length,
                    /* trans= */ false,
                    /* useGpu= */ false,
                    /* useWeight= */ false,
                    epsilon());
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
