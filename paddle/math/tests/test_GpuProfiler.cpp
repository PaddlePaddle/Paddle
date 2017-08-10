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

#ifndef PADDLE_ONLY_CPU

#include <gtest/gtest.h>
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"
#include "paddle/testing/TestUtil.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

void MatrixCheckErr(const Matrix& matrix1, const Matrix& matrix2) {
  CHECK(matrix1.getHeight() == matrix2.getHeight());
  CHECK(matrix1.getWidth() == matrix2.getWidth());
#ifndef PADDLE_TYPE_DOUBLE
  real err = 1e-3;
#else
  real err = 1e-10;
#endif

  int height = matrix1.getHeight();
  int width = matrix1.getWidth();
  const real* data1 = matrix1.getData();
  const real* data2 = matrix2.getData();
  int count = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      real a = data1[i * width + j];
      real b = data2[i * width + j];
      if (fabs(a - b) > err) {
        if ((fabsf(a - b) / fabsf(a)) > (err / 10.0f)) {
          count++;
        }
      }
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different element.";
}

void testBilinearFwdBwd(int numSamples,
                        int imgSizeH,
                        int imgSizeW,
                        int channels) {
  int inWidth = imgSizeH * imgSizeW * channels;
  int outWidth = 2 * imgSizeH * 2 * imgSizeW * channels;
  real ratioH = 0.5;
  real ratioW = 0.5;

  // forward
  MatrixPtr input = CpuMatrix::create(numSamples, inWidth, false, false);
  MatrixPtr inputGpu = GpuMatrix::create(numSamples, inWidth, false, true);

  MatrixPtr target = CpuMatrix::create(numSamples, outWidth, false, false);
  MatrixPtr targetGpu = GpuMatrix::create(numSamples, outWidth, false, true);
  MatrixPtr targetCheck = CpuMatrix::create(numSamples, outWidth, false, false);

  input->randomizeUniform();
  inputGpu->copyFrom(*input);

  {
    // nvprof: GPU Proflier
    REGISTER_GPU_PROFILER("testBilinearFwdBwd");
    target->bilinearForward(*input,
                            imgSizeH,
                            imgSizeW,
                            2 * imgSizeH,
                            2 * imgSizeW,
                            channels,
                            ratioH,
                            ratioW);
    targetGpu->bilinearForward(*inputGpu,
                               imgSizeH,
                               imgSizeW,
                               2 * imgSizeH,
                               2 * imgSizeW,
                               channels,
                               ratioH,
                               ratioW);
  }

  // check
  targetCheck->copyFrom(*targetGpu);
  MatrixCheckErr(*target, *targetCheck);

  // backward
  MatrixPtr inputGrad = CpuMatrix::create(numSamples, inWidth, false, false);
  MatrixPtr inputGpuGrad = GpuMatrix::create(numSamples, inWidth, false, true);

  MatrixPtr targetGrad = CpuMatrix::create(numSamples, outWidth, false, false);
  MatrixPtr targetGpuGrad =
      GpuMatrix::create(numSamples, outWidth, false, true);
  MatrixPtr targetCheckGrad =
      CpuMatrix::create(numSamples, inWidth, false, false);

  inputGrad->randomizeUniform();
  targetGrad->randomizeUniform();
  inputGpuGrad->copyFrom(*inputGrad);
  targetGpuGrad->copyFrom(*targetGrad);

  inputGrad->bilinearBackward(*targetGrad,
                              2 * imgSizeH,
                              2 * imgSizeW,
                              imgSizeH,
                              imgSizeW,
                              channels,
                              ratioH,
                              ratioW);
  inputGpuGrad->bilinearBackward(*targetGpuGrad,
                                 2 * imgSizeH,
                                 2 * imgSizeW,
                                 imgSizeH,
                                 imgSizeW,
                                 channels,
                                 ratioH,
                                 ratioW);

  // check
  targetCheckGrad->copyFrom(*inputGpuGrad);
  MatrixCheckErr(*inputGrad, *targetCheckGrad);
}

TEST(Profiler, testBilinearFwdBwd) {
  auto numSamples = 10;
  auto channels = 16;
  auto imgSize = 64;
  {
    // nvprof: GPU Proflier
    REGISTER_GPU_PROFILER("testBilinearFwdBwd");
    // Paddle built-in timer
    REGISTER_TIMER_INFO(
        "testBilinearFwdBwd",
        "numSamples = 10, channels = 16, imgSizeX = 64, imgSizeY = 64");
    testBilinearFwdBwd(numSamples, imgSize, imgSize, channels);
  }
  globalStat.printAllStatus();
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);

  // nvprof: GPU Proflier
  REGISTER_GPU_PROFILER(
      "RecursiveProfilingTest",
      "numSamples = 10, channels = 16, imgSizeX = 64, imgSizeY = 64");

  return RUN_ALL_TESTS();
}

#endif /* PADDLE_ONLY_CPU */
