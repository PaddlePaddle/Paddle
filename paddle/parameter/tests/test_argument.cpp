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
#include <paddle/parameter/Argument.h>

using namespace paddle;  // NOLINT

TEST(Argument, poolSequenceWithStride) {
  Argument input, output;
  ICpuGpuVector::resizeOrCreate(input.sequenceStartPositions, 5, false);
  int* inStart = input.sequenceStartPositions->getMutableData(false);
  inStart[0] = 0;
  inStart[1] = 9;
  inStart[2] = 14;
  inStart[3] = 17;
  inStart[4] = 30;

  int strideResult[] = {0, 5, 9, 14, 17, 22, 27, 30};
  int strideResultReversed[] = {0, 4, 9, 14, 17, 20, 25, 30};

  for (auto reversed : {false, true}) {
    IVectorPtr stridePositions;
    output.poolSequenceWithStride(
        input, 5 /* stride */, &stridePositions, reversed);

    const int* outStart = output.sequenceStartPositions->getData(false);
    CHECK_EQ(outStart[0], 0);
    CHECK_EQ(outStart[1], 2);
    CHECK_EQ(outStart[2], 3);
    CHECK_EQ(outStart[3], 4);
    CHECK_EQ(outStart[4], 7);

    CHECK_EQ(stridePositions->getSize(), 8);
    auto result = reversed ? strideResultReversed : strideResult;
    for (int i = 0; i < 8; i++) {
      CHECK_EQ(stridePositions->getData()[i], result[i]);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  return RUN_ALL_TESTS();
}
