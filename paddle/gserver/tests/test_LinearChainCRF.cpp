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
#include <vector>
#include "paddle/gserver/layers/LinearChainCRF.h"
#include "paddle/utils/Util.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

static inline bool getNextSequence(vector<int>& seq, int numClasses) {
  for (auto& v : seq) {
    if (++v < numClasses) {
      return true;
    }
    v = 0;
  }
  return false;
}

TEST(LinearChainCRF, decoding) {
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
      vector<int> decodingResult(length);
      vector<int> bestResult(length);
      vector<int> testResult(length, 0);
      crf.decode(x.getData(), &decodingResult[0], length);
      real bestScore = -std::numeric_limits<real>::max();
      do {
        real score = a[testResult.front()] + b[testResult.back()];
        score += x.getElement(0, testResult.front());
        for (int k = 1; k < length; ++k) {
          score += x.getElement(k, testResult[k]) +
                   w[numClasses * testResult[k - 1] + testResult[k]];
        }
        if (score > bestScore) {
          bestScore = score;
          bestResult = testResult;
        }
      } while (getNextSequence(testResult, numClasses));
      for (int k = 0; k < length; ++k) {
        EXPECT_EQ(decodingResult[k], bestResult[k]);
      }
    }
  }
}
