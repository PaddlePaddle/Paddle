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
/**
 * This test file use autotest::AutoCompare and cmpWithArg to compares the
 * implementation of CPU and GPU member function in Matrix.cpp.
 */

#include <gtest/gtest.h>
#include "TestUtils.h"

using paddle::BaseMatrix;
using paddle::Matrix;
using paddle::CpuMatrix;
using paddle::CpuIVector;
using paddle::CpuSparseMatrix;
using autotest::AutoCompare;

void testBilinearFwdBwd(int numSamples,
                        int imgSizeH,
                        int imgSizeW,
                        int channels) {
  int inWidth = imgSizeH * imgSizeW * channels;
  int outWidth = 2 * imgSizeH * 2 * imgSizeW * channels;
  real ratioH = 0.5;
  real ratioW = 0.5;

  AutoCompare forward(numSamples, outWidth);
  CpuMatrix arg1(numSamples, inWidth);
  arg1.randomizeUniform();
  forward.cmpWithArg(&Matrix::bilinearForward,
                     arg1,
                     imgSizeH,
                     imgSizeW,
                     2 * imgSizeH,
                     2 * imgSizeW,
                     channels,
                     ratioH,
                     ratioW);

  AutoCompare backward(numSamples, inWidth);
  CpuMatrix arg2(numSamples, outWidth);
  arg2.randomizeUniform();
  backward.cmpWithArg(&Matrix::bilinearBackward,
                      arg2,
                      2 * imgSizeH,
                      2 * imgSizeW,
                      imgSizeH,
                      imgSizeW,
                      channels,
                      ratioH,
                      ratioW);
}

TEST(Matrix, BilinearFwdBwd) {
  for (auto numSamples : {5, 10}) {
    for (auto channels : {8, 16}) {
      for (auto imgSizeH : {14, 28}) {
        for (auto imgSizeW : {16, 30}) {
          VLOG(3) << " numSamples=" << numSamples << " channels=" << channels
                  << " imgSizeH=" << imgSizeH << " imgSizeW=" << imgSizeW;
          testBilinearFwdBwd(numSamples, imgSizeH, imgSizeW, channels);
        }
      }
    }
  }
}

void testMatrixAddBias(int height, int width, real scale) {
  AutoCompare test(height, width);
  CpuMatrix arg1(1, width);
  arg1.randomizeUniform();
  test.cmpWithArg(
      static_cast<void (Matrix::*)(Matrix&, real)>(&Matrix::addBias),
      arg1,
      scale);
}

void testMatrixAddDotMulMMV(int height, int width) {
  AutoCompare test(height, width);
  CpuMatrix arg1(height, width);
  CpuMatrix arg2(1, width);
  arg1.randomizeUniform();
  arg2.randomizeUniform();
  test.cmpWithArg(&BaseMatrix::addDotMulMMV, arg1, arg2);
}

TEST(Matrix, unary) {
  for (auto height : {1, 3, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 3, 32, 100, 512, 1000, 3210}) {
      VLOG(3) << " height=" << height << " width=" << width;
      testMatrixAddBias(height, width, 1.0);
      testMatrixAddBias(height, width, 3.5);
      testMatrixAddDotMulMMV(height, width);
    }
  }
}

void testMatrixAddAtOffset(int height, int width1, int width2, int offset) {
  AutoCompare test(height, width2);
  CpuMatrix arg1(height, width1);
  arg1.randomizeUniform();
  test.cmpWithArg(&Matrix::addAtOffset, arg1, offset);
}

void testMatrixAssignAtOffset(int height, int width1, int width2, int offset) {
  AutoCompare test(height, width2);
  CpuMatrix arg1(height, width1);
  arg1.randomizeUniform();
  test.cmpWithArg(&Matrix::assignAtOffset, arg1, offset);
}

TEST(Matrix, AtOffset) {
  for (auto height : {1, 11, 73, 128, 200}) {
    for (auto width1 : {1, 32, 100, 512, 1000}) {
      for (auto width2 : {1, 32, 100, 512, 1000}) {
        int columnOffset = 0;
        int offset = std::abs(width1 - width2);
        if (offset) {
          columnOffset = std::rand() % offset;
        }
        VLOG(3) << " height=" << height << " width1=" << width1
                << " width2=" << width2 << " columnOffset = " << columnOffset;
        testMatrixAddAtOffset(height, width1, width2, columnOffset);
        testMatrixAssignAtOffset(height, width1, width2, columnOffset);
      }
    }
  }
}

void testMatrixSelectRows(int numSamples, int tableSize, int inputDim) {
  AutoCompare test(numSamples, inputDim);
  CpuMatrix arg1(tableSize, inputDim);
  CpuIVector arg2(numSamples);
  arg1.randomizeUniform();
  arg2.rand(tableSize);
  test.cmpWithArg(&Matrix::selectRows, arg1, arg2);
}

TEST(Matrix, tableProjection) {
  for (auto numSamples : {10, 100, 1000, 10000, 80000}) {
    for (auto tableSize : {10, 100}) {
      for (auto inputDim : {20, 50}) {
        VLOG(3) << " numSamples=" << numSamples << " tableSize=" << tableSize
                << " inputDim=" << inputDim;
        testMatrixSelectRows(numSamples, tableSize, inputDim);
      }
    }
  }
}

void testMatrixCopyByRowIndex(int outHeight, int inHeight, int width) {
  AutoCompare test(outHeight, width);
  CpuMatrix arg1(inHeight, width);
  CpuIVector arg2(outHeight);
  arg1.randomizeUniform();
  arg2.rand(inHeight);
  test.cmpWithArg(&Matrix::copyByRowIndex, arg1, arg2);
}

TEST(Matrix, copyByRowIndex) {
  for (auto outHeight : {31, 500, 1000}) {
    for (auto inHeight : {17, 257, 500, 1200}) {
      for (auto width : {512, 1024}) {
        VLOG(3) << outHeight << " " << inHeight << " " << width;
        testMatrixCopyByRowIndex(outHeight, inHeight, width);
      }
    }
  }
}

void testParamReluForward(int height, int width, int w_height, int w_width) {
  AutoCompare test(height, width);
  CpuMatrix arg1(height, width);
  CpuMatrix arg2(w_height, w_width);
  arg1.randomizeUniform();
  arg2.randomizeUniform();
  arg1.add(-0.5);
  test.cmpWithArg(&Matrix::paramReluForward, arg1, arg2);
}

void testParamReluBackwardW(int height, int width, int w_height, int w_width) {
  AutoCompare test(w_height, w_width);
  CpuMatrix arg1(height, width);
  CpuMatrix arg2(height, width);
  arg1.randomizeUniform();
  arg2.randomizeUniform();
  arg2.add(-0.5);
  test.cmpWithArg(&Matrix::paramReluBackwardW, arg1, arg2);
}

TEST(Matrix, paramRelu) {
  for (auto height : {10, 40, 100}) {
    for (auto width : {10, 40, 100}) {
      for (auto w_height : {1, 2}) {
        for (auto w_width : {1, 2}) {
          if (width % (w_height * w_width)) continue;
          testParamReluForward(height, width, w_height, w_width);
          testParamReluBackwardW(height, width, w_height, w_width);
        }
      }
    }
  }
}

void testAddSharedBias(int numSamples, int dim, int channel) {
  AutoCompare test(numSamples, dim);
  CpuMatrix arg1(1, channel);
  arg1.randomizeUniform();
  test.cmpWithArg(&Matrix::addSharedBias, arg1, 1.0);
}

void testCollectSharedBias(int numSamples, int dim, int channel) {
  AutoCompare test(1, channel);
  CpuMatrix arg1(numSamples, dim);
  arg1.randomizeUniform();
  test.cmpWithArg(&Matrix::collectSharedBias, arg1, 1.0);
}

TEST(Matrix, sharedBias) {
  for (auto numSamples : {1, 100, 520}) {
    for (auto dim : {100 * 16, 100 * 32}) {
      for (auto channel : {8, 16}) {
        VLOG(3) << " numSamples=" << numSamples << " dim=" << dim
                << " channel=" << channel;
        testAddSharedBias(numSamples, dim, channel);
        testCollectSharedBias(numSamples, dim, channel);
      }
    }
  }
}

void testMultiBinaryLabelCrossEntropy(int numSamples, int dim) {
  AutoCompare forward(numSamples, 1);
  CpuMatrix arg1(numSamples, dim);
  CpuSparseMatrix arg2(
      numSamples, dim, numSamples, paddle::NO_VALUE, paddle::SPARSE_CSR);

  CpuMatrix output1(numSamples, dim);
  output1.randomizeUniform();
  output1.softmax(arg1);
  for (int i = 0; i < numSamples; i++) {
    const unsigned int id = std::rand() % dim;
    arg2.setRow(i, 1, &id, nullptr);
  }
  forward.cmpWithArg(&Matrix::multiBinaryLabelCrossEntropy, arg1, arg2);

  AutoCompare backward(numSamples, dim);
  backward.cmpWithArg(&Matrix::multiBinaryLabelCrossEntropyBp, arg1, arg2);
}

TEST(Matrix, multiBinaryCrossEntropy) {
  for (auto numSamples : {100, 1000, 10000}) {
    for (auto dim : {100, 1000, 10000}) {
      VLOG(3) << " numSamples=" << numSamples << " dim=" << dim;
      testMultiBinaryLabelCrossEntropy(numSamples, dim);
    }
  }
}

#endif
