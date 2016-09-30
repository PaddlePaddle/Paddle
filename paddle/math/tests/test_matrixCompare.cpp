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

#ifndef PADDLE_ONLY_CPU
/// This unittest checks GpuMatrix/CpuMatrix get same result, so disable when
/// only cpu version.

#include "paddle/utils/Util.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"
#include <gtest/gtest.h>
#include "paddle/gserver/tests/TestUtil.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

template<class T>
void VectorCheckEqual(const VectorT<T>& vector1, const VectorT<T>& vector2) {
  CHECK(vector1.getSize() == vector2.getSize());

  const T* data1 = vector1.getData();
  const T* data2 = vector2.getData();
  size_t size = vector1.getSize();
  int count = 0;
  for (size_t i = 0; i < size; i++) {
    if (data1[i] != data2[i]) {
      count++;
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different element.";
}

void MatrixCheckEqual(const Matrix& matrix1, const Matrix& matrix2) {
  CHECK(matrix1.getHeight() == matrix2.getHeight());
  CHECK(matrix1.getWidth() == matrix2.getWidth());

  int height = matrix1.getHeight();
  int width = matrix1.getWidth();
  const real* data1 = matrix1.getData();
  const real* data2 = matrix2.getData();
  int count = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (data1[i * width + j] != data2[i * width + j]) {
        count++;
      }
    }
  }
  EXPECT_EQ(count, 0) << "There are " << count << " different element.";
}

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

void testMatrixProjectionForward(int contextStart, int contextLength,
                                 bool padding, int batchSize, int inputDim) {
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(batchSize, inputDim);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(batchSize, inputDim);
  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);

  int pad = std::max(0, -contextStart) +
            std::max(0, contextStart + contextLength - 1);
  if (pad == 0) padding = false;
  MatrixPtr cpuWeight = nullptr;
  MatrixPtr gpuWeight = nullptr;
  if (padding) {
    cpuWeight = std::make_shared<CpuMatrix>(pad, inputDim);
    gpuWeight = std::make_shared<GpuMatrix>(pad, inputDim);
    cpuWeight->randomizeUniform();
    gpuWeight->copyFrom(*cpuWeight);
  }

  IVectorPtr cpuSequence;
  generateSequenceStartPositions(batchSize, cpuSequence);
  IVectorPtr gpuSequence = IVector::create(cpuSequence->getSize(), true);
  gpuSequence->copyFrom(*cpuSequence);

  MatrixPtr cpuOutput =
      std::make_shared<CpuMatrix>(batchSize, inputDim * contextLength);
  MatrixPtr gpuOutput =
      std::make_shared<GpuMatrix>(batchSize, inputDim * contextLength);
  cpuOutput->randomizeUniform();
  gpuOutput->copyFrom(*cpuOutput);

  // calculate
  int beginPad = std::max(0, -contextStart);
  cpuOutput->contextProjectionForward(cpuInput, cpuWeight, *cpuSequence,
                                      contextLength, contextStart, beginPad,
                                      padding);

  gpuOutput->contextProjectionForward(gpuInput, gpuWeight, *gpuSequence,
                                      contextLength, contextStart, beginPad,
                                      padding);

  // check
  MatrixPtr outputCheck =
      std::make_shared<CpuMatrix>(batchSize, inputDim * contextLength);
  outputCheck->copyFrom(*gpuOutput);

  MatrixCheckEqual(*cpuOutput, *outputCheck);
}

void testMatrixProjectionBackward(int contextStart, int contextLength,
                                  bool padding, int batchSize, int inputDim) {
  MatrixPtr cpuOutputGrad =
      std::make_shared<CpuMatrix>(batchSize, inputDim * contextLength);
  MatrixPtr gpuOutputGrad =
      std::make_shared<GpuMatrix>(batchSize, inputDim * contextLength);
  cpuOutputGrad->randomizeUniform();
  gpuOutputGrad->copyFrom(*cpuOutputGrad);

  IVectorPtr cpuSequence;
  generateSequenceStartPositions(batchSize, cpuSequence);
  IVectorPtr gpuSequence = IVector::create(cpuSequence->getSize(), true);
  gpuSequence->copyFrom(*cpuSequence);

  MatrixPtr cpuInputGrad = std::make_shared<CpuMatrix>(batchSize, inputDim);
  MatrixPtr gpuInputGrad = std::make_shared<GpuMatrix>(batchSize, inputDim);
  cpuInputGrad->randomizeUniform();
  gpuInputGrad->copyFrom(*cpuInputGrad);

  int pad = std::max(0, -contextStart) +
            std::max(0, contextStart + contextLength - 1);
  if (pad == 0) padding = false;
  MatrixPtr cpuWeightGrad = nullptr;
  MatrixPtr gpuWeightGrad = nullptr;
  if (padding) {
    cpuWeightGrad = std::make_shared<CpuMatrix>(pad, inputDim);
    gpuWeightGrad = std::make_shared<GpuMatrix>(pad, inputDim);
    cpuWeightGrad->randomizeUniform();
    gpuWeightGrad->copyFrom(*cpuWeightGrad);
  }

  // calculate
  int beginPad = std::max(0, -contextStart);
  cpuOutputGrad->contextProjectionBackward(cpuInputGrad, cpuWeightGrad,
                                           *cpuSequence, contextLength,
                                           contextStart, beginPad, padding);
  gpuOutputGrad->contextProjectionBackwardData(gpuInputGrad, *gpuSequence,
                                               contextLength, contextStart);
  if (padding) {
    gpuOutputGrad->contextProjectionBackwardWeight(
        gpuWeightGrad, *gpuSequence, contextLength,
        contextStart, pad, beginPad);
  }

  // check
  MatrixPtr inputGradCheck = std::make_shared<CpuMatrix>(batchSize, inputDim);
  inputGradCheck->copyFrom(*gpuInputGrad);
  MatrixCheckErr(*cpuInputGrad, *inputGradCheck);

  if (padding) {
    MatrixPtr weightGradChcek = std::make_shared<CpuMatrix>(pad, inputDim);
    weightGradChcek->copyFrom(*gpuWeightGrad);
    MatrixCheckErr(*cpuWeightGrad, *weightGradChcek);
  }
}

TEST(Matrix, projection) {
  for (auto contextStart : {-5, -3, -1, 0, 3}) {
    for (auto contextLength : {1, 2, 5, 7}) {
      for (auto trainablePadding : {false, true}) {
        for (auto batchSize : {1, 2, 5, 20, 100}) {
          for (auto inputDim : {15, 32, 63, 128, 200}) {
            VLOG(3) << " contextStart=" << contextStart
                      << " contextLength=" << contextLength
                      << " trainablePadding=" << trainablePadding
                      << " batchSize=" << batchSize << " inputDim=" << inputDim;
            testMatrixProjectionForward(contextStart, contextLength,
                                        trainablePadding, batchSize, inputDim);
            testMatrixProjectionBackward(contextStart, contextLength,
                                         trainablePadding, batchSize, inputDim);
          }
        }
      }
    }
  }
}

void testMatrixMaxSequence(int batchSize, int inputDim) {
  // forward
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(batchSize, inputDim);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(batchSize, inputDim);
  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);

  IVectorPtr cpuSequence;
  generateSequenceStartPositions(batchSize, cpuSequence);
  IVectorPtr gpuSequence = IVector::create(cpuSequence->getSize(), true);
  gpuSequence->copyFrom(*cpuSequence);

  int newBatchSize = cpuSequence->getSize() - 1;
  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(newBatchSize, inputDim);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(newBatchSize, inputDim);
  cpuOutput->zero();
  gpuOutput->zero();

  IVectorPtr cpuIndex = nullptr;
  IVectorPtr gpuIndex = nullptr;
  IVector::resizeOrCreate(cpuIndex, newBatchSize * inputDim, false);
  IVector::resizeOrCreate(gpuIndex, newBatchSize * inputDim, true);
  cpuIndex->zeroMem();
  gpuIndex->zeroMem();

  cpuOutput->maxSequenceForward(*cpuInput, *cpuSequence, *cpuIndex);
  gpuOutput->maxSequenceForward(*gpuInput, *gpuSequence, *gpuIndex);

  // check
  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(newBatchSize, inputDim);
  outputCheck->copyFrom(*gpuOutput);
  MatrixCheckEqual(*cpuOutput, *outputCheck);

  IVectorPtr indexCheck = nullptr;
  IVector::resizeOrCreate(indexCheck, newBatchSize * inputDim, false);
  indexCheck->copyFrom(*gpuIndex);
  VectorCheckEqual(*cpuIndex, *indexCheck);

  // backward
  MatrixPtr cpuOutputGrad = std::make_shared<CpuMatrix>(newBatchSize, inputDim);
  MatrixPtr gpuOutputGrad = std::make_shared<GpuMatrix>(newBatchSize, inputDim);
  cpuOutputGrad->randomizeUniform();
  gpuOutputGrad->copyFrom(*cpuOutputGrad);

  MatrixPtr cpuInputGrad = std::make_shared<CpuMatrix>(batchSize, inputDim);
  MatrixPtr gpuInputGrad = std::make_shared<GpuMatrix>(batchSize, inputDim);
  cpuInputGrad->randomizeUniform();
  gpuInputGrad->copyFrom(*cpuInputGrad);

  cpuInputGrad->maxSequenceBackward(*cpuOutputGrad, *cpuSequence, *cpuIndex);
  gpuInputGrad->maxSequenceBackward(*gpuOutputGrad, *gpuSequence, *gpuIndex);

  // check
  MatrixPtr inputGradCheck = std::make_shared<CpuMatrix>(batchSize, inputDim);
  inputGradCheck->copyFrom(*gpuInputGrad);
  MatrixCheckEqual(*cpuInputGrad, *inputGradCheck);
}

TEST(Matrix, maxSequence) {
  for (auto batchSize : {1, 10, 128, 1000, 6000}) {
    for (auto inputDim : {1, 32, 100, 512}) {
      VLOG(3) << " batchSize=" << batchSize << " inputDim=" << inputDim;
      testMatrixMaxSequence(batchSize, inputDim);
    }
  }
}

void testMatrixGetSum(int height, int width) {
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(height, width);
  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);

#ifndef PADDLE_TYPE_DOUBLE
  int x = log10(height * width);
  real err = 1e-6 * pow(10, x);
#else
  real err = 1e-8;
#endif

  real cpuSum = cpuInput->getSum();
  real gpuSum = gpuInput->getSum();

  EXPECT_LE(fabs(cpuSum - gpuSum), err);
}

void testMatrixZeroAtOffset(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr cpuTest = std::make_shared<CpuMatrix>(height, width);

  cpuA->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  cpuTest->copyFrom(*cpuA);

  int columnOffset = rand() % width;  // NOLINT we just use rand() for test.
  int numColumns = rand() % (width - columnOffset);  // NOLINT

  cpuA->zeroAtOffset(columnOffset, numColumns);
  gpuA->zeroAtOffset(columnOffset, numColumns);

  /* cpuTest */
  real* a = cpuTest->getData() + columnOffset;
  for (int64_t i = 0; i < height; ++i) {
    for (int64_t j = 0; j < numColumns; ++j) {
      a[i * width + j] = 0;
    }
  }

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckEqual(*cpuA, *outputCheck);
  MatrixCheckEqual(*cpuA, *cpuTest);
}

void testMatrixBinaryAdd(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  cpuA->add(*cpuB);
  gpuA->add(*gpuB);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckEqual(*cpuA, *outputCheck);
}

void testMatrixAssign(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  cpuA->assign(2.5);
  gpuA->assign(2.5);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckEqual(*cpuA, *outputCheck);
}

void testMatrixAdd(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  cpuA->add(2.5);
  gpuA->add(2.5);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckEqual(*cpuA, *outputCheck);
}

void testMatrixSqrt(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  cpuA->sqrt2();
  gpuA->sqrt2();

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixTanhDerivative(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  cpuA->tanhDerivative(*cpuB);
  gpuA->tanhDerivative(*gpuB);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixTanh(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  cpuA->tanh(*cpuB);
  gpuA->tanh(*gpuB);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixTernarySub(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuC = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);

  cpuA->sub(*cpuB, *cpuC);
  gpuA->sub(*gpuB, *gpuC);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckEqual(*cpuA, *outputCheck);
}

void testMatrixSumOfSquaresBp(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuC = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);

  cpuA->sumOfSquaresBp(*cpuB, *cpuC);
  gpuA->sumOfSquaresBp(*gpuB, *gpuC);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixBinaryRowScale(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, 1);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, 1);

  MatrixPtr cpuA1 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB1 = std::make_shared<CpuMatrix>(height, 1);
  MatrixPtr gpuA1 = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB1 = std::make_shared<GpuMatrix>(height, 1);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  cpuA1->copyFrom(*cpuA);
  cpuB1->copyFrom(*cpuB);
  gpuA1->copyFrom(*cpuA);
  gpuB1->copyFrom(*cpuB);

  cpuA->addColVector(*cpuB);
  gpuA->addColVector(*gpuB);
  cpuA1->addColumnVector(*cpuB1);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckEqual(*cpuA, *outputCheck);

  MatrixCheckEqual(*cpuA, *cpuA1);
}

void testMatrixAddBias(int height, int width, real scale) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(1, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(1, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);

  cpuA->addBias(*cpuB, scale);
  gpuA->addBias(*gpuB, scale);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixTernaryRowScale(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuC = std::make_shared<GpuMatrix>(height, width);

  MatrixPtr cpuA1 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB1 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC1 = std::make_shared<CpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);
  cpuA1->copyFrom(*cpuA);
  cpuB1->copyFrom(*cpuB);
  cpuC1->copyFrom(*cpuC);

  int columnOffset = rand() % width;  // NOLINT

  cpuA->rowScale(columnOffset, *cpuB, *cpuC);
  gpuA->rowScale(columnOffset, *gpuB, *gpuC);
  cpuA1->rowScale2(columnOffset, *cpuB1, *cpuC1);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckEqual(*cpuA, *outputCheck);

  MatrixCheckEqual(*cpuA, *cpuA1);
}

void testMatrixTernaryRowDotMul(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC = std::make_shared<CpuMatrix>(height, width);

  MatrixPtr cpuA1 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB1 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC1 = std::make_shared<CpuMatrix>(height, width);

  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuC = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  cpuA1->copyFrom(*cpuA);
  cpuB1->copyFrom(*cpuB);
  cpuC1->copyFrom(*cpuC);
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);

  int columnOffset = rand() % width;  // NOLINT

  cpuA->rowDotMul(columnOffset, *cpuB, *cpuC);
  gpuA->rowDotMul(columnOffset, *gpuB, *gpuC);
  cpuA1->rowDotMul2(columnOffset, *cpuB1, *cpuC1);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *cpuA1);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixAddDotMulMMV(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC = std::make_shared<CpuMatrix>(1, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuC = std::make_shared<GpuMatrix>(1, width);

  MatrixPtr cpuA1 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB1 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC1 = std::make_shared<CpuMatrix>(1, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);
  cpuA1->copyFrom(*cpuA);
  cpuB1->copyFrom(*cpuB);
  cpuC1->copyFrom(*cpuC);

  cpuA->addDotMulMMV(*cpuB, *cpuC);
  gpuA->addDotMulMMV(*gpuB, *gpuC);
  cpuA1->addDotMulMMV2(*cpuB1, *cpuC1);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
  MatrixCheckEqual(*cpuA, *cpuA1);
}

void testMatrixTranspose(int height, int width) {
  MatrixPtr cpu = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpu = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr cpuT = std::make_shared<CpuMatrix>(width, height);
  MatrixPtr gpuT = std::make_shared<GpuMatrix>(width, height);

  cpu->randomizeUniform();
  gpu->copyFrom(*cpu);
  cpu->transpose(cpuT, false);
  gpu->transpose(gpuT, false);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(width, height);
  outputCheck->copyFrom(*gpuT);
  MatrixCheckEqual(*cpuT, *outputCheck);
}

TEST(Matrix, unary) {
  for (auto height : {1, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 32, 100, 512, 1000, 3210}) {
      VLOG(3) << " height=" << height << " width=" << width;

      // applyUnary
      testMatrixAssign(height, width);
      testMatrixAdd(height, width);
      testMatrixSqrt(height, width);

      // applyBinary
      testMatrixBinaryAdd(height, width);
      testMatrixTanh(height, width);
      testMatrixTanhDerivative(height, width);

      // applyTernary
      testMatrixTernarySub(height, width);
      testMatrixSumOfSquaresBp(height, width);

      // asRowVector
      testMatrixAddBias(height, width, 1.0);
      testMatrixAddBias(height, width, 3.5);
      testMatrixAddDotMulMMV(height, width);

      // asColVector
      testMatrixTernaryRowScale(height, width);
      testMatrixBinaryRowScale(height, width);

      // sum
      testMatrixGetSum(height, width);

      // transpose
      testMatrixTranspose(height, width);
    }
  }
}

void testMatrixSoftmax(int height, int width) {
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(height, width);

  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);
  cpuOutput->zero();
  gpuOutput->zero();
  cpuInput->softmax(*cpuOutput);
  gpuInput->softmax(*gpuOutput);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuOutput);
  MatrixCheckErr(*cpuOutput, *outputCheck);
}

void testSequenceSoftmax(int batchSize) {
  // forward
  int inputDim = 1;
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(batchSize, inputDim);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(batchSize, inputDim);
  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);

  IVectorPtr cpuSequence;
  generateSequenceStartPositions(batchSize, cpuSequence);
  IVectorPtr gpuSequence = IVector::create(cpuSequence->getSize(), true);
  gpuSequence->copyFrom(*cpuSequence);

  cpuInput->sequenceSoftmax(*cpuInput, *cpuSequence);
  gpuInput->sequenceSoftmax(*gpuInput, *gpuSequence);

  // check
  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(batchSize, inputDim);
  outputCheck->copyFrom(*gpuInput);
  MatrixCheckErr(*cpuInput, *outputCheck);
}


void testMatrixSoftmaxThreshold(int height, int width) {
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(height, width);

  cpuInput->randomizeUniform();
  cpuInput->getData()[0] = 100.0;
  gpuInput->copyFrom(*cpuInput);
  cpuOutput->zero();
  gpuOutput->zero();
  cpuInput->softmax(*cpuOutput);
  gpuInput->softmax(*gpuOutput);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuOutput);
  // check output zero
  int cpuCount = 0;
  int gpuCount = 0;
  auto zeroNum = [](MatrixPtr out, int& count) {
    for (size_t i = 0; i < out->getHeight(); i++) {
      for (size_t j = 0; j < out->getWidth(); j++) {
        if (out->getElement(i, j) == 0) count++;
      }
    }
  };
  zeroNum(cpuOutput, cpuCount);
  zeroNum(outputCheck, gpuCount);
  EXPECT_EQ(cpuCount, 0) << "Cpu softmax output value 0";
  EXPECT_EQ(gpuCount, 0) << "Gpu softmax output value 0";
}

void testMatrixSoftmaxBp(int height, int width) {
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(height, width);

  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);
  cpuOutput->randomizeUniform();
  gpuOutput->copyFrom(*cpuOutput);
  gpuOutput->softmaxBackward(*gpuInput);

  MatrixPtr sftMaxSum = std::make_shared<CpuMatrix>(height, 1);
  MatrixPtr sftMaxDot = std::make_shared<CpuMatrix>(height, width);
  sftMaxDot->dotMul(*cpuOutput, *cpuInput);
  sftMaxSum->colMerge(*sftMaxDot);
  cpuOutput->softmaxDerivative(*cpuInput, *sftMaxSum);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuOutput);
  MatrixCheckErr(*cpuOutput, *outputCheck);
}

TEST(Matrix, softmax) {
  for (auto height : {1, 11, 73, 128, 200}) {
    for (auto width : {1, 32, 100, 512, 1000}) {
      VLOG(3) << " height=" << height << " width=" << width;

      testMatrixSoftmax(height, width);
      testMatrixSoftmaxBp(height, width);
      testMatrixSoftmaxThreshold(height, width);
    }
    testSequenceSoftmax(height);
  }
}

void testMatrixAddDotMulVMM(int height, int width, int endCol = 0) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(1, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(1, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuC = std::make_shared<GpuMatrix>(height, width);

  MatrixPtr cpuA1 = std::make_shared<CpuMatrix>(1, width);
  MatrixPtr cpuB1 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC1 = std::make_shared<CpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);
  cpuA1->copyFrom(*cpuA);
  cpuB1->copyFrom(*cpuB);
  cpuC1->copyFrom(*cpuC);

  if (!endCol) {
    cpuA->addDotMulVMM(*cpuB, *cpuC);
    gpuA->addDotMulVMM(*gpuB, *gpuC);
    cpuA1->addDotMulVMM2(*cpuB1, *cpuC1);

    MatrixCheckErr(*cpuA, *cpuA1);
  } else {
    MatrixPtr subCpuA = cpuA->subColMatrix(0, endCol);
    MatrixPtr subCpuB = cpuB->subColMatrix(0, endCol);
    MatrixPtr subCpuC = cpuC->subColMatrix(0, endCol);
    MatrixPtr subGpuA = gpuA->subColMatrix(0, endCol);
    MatrixPtr subGpuB = gpuB->subColMatrix(0, endCol);
    MatrixPtr subGpuC = gpuC->subColMatrix(0, endCol);
    subCpuA->addDotMulVMM(*subCpuB, *subCpuC);
    subGpuA->addDotMulVMM(*subGpuB, *subGpuC);
  }

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(1, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixRowSum(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, 1);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, 1);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);

  MatrixPtr cpuA1 = std::make_shared<CpuMatrix>(height, 1);
  MatrixPtr cpuB1 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA1 = std::make_shared<GpuMatrix>(height, 1);
  MatrixPtr gpuB1 = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  cpuA1->copyFrom(*cpuA);
  cpuB1->copyFrom(*cpuB);
  gpuA1->copyFrom(*cpuA);
  gpuB1->copyFrom(*cpuB);

  cpuA->colMerge(*cpuB);
  gpuA->colMerge(*gpuB);

  cpuB1->rowSum(*cpuA1);
  gpuB1->rowSum(*gpuA1);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, 1);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
  outputCheck->copyFrom(*gpuA1);
  MatrixCheckErr(*cpuA1, *outputCheck);
}

void testMatrixRowMax(int height, int width, int endCol = 0) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, 1);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, 1);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);

  if (!endCol) {
    cpuB->rowMax(*cpuA);
    gpuB->rowMax(*gpuA);
  } else {
    MatrixPtr subCpuB = cpuB->subColMatrix(0, endCol);
    MatrixPtr subGpuB = gpuB->subColMatrix(0, endCol);
    subCpuB->rowMax(*cpuA);
    subGpuB->rowMax(*gpuA);
  }

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, 1);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixColSum(int height, int width, int endCol = 0) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(1, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(1, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);

  if (!endCol) {
    cpuA->accumulateColSum(*cpuB);
    gpuA->accumulateColSum(*gpuB);
  } else {
    MatrixPtr subCpuA = cpuA->subColMatrix(0, endCol);
    MatrixPtr subGpuA = gpuA->subColMatrix(0, endCol);
    MatrixPtr subCpuB = cpuB->subColMatrix(0, endCol);
    MatrixPtr subGpuB = gpuB->subColMatrix(0, endCol);
    subCpuA->accumulateColSum(*subCpuB);
    subGpuA->accumulateColSum(*subGpuB);
  }

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(1, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixColMax(int height, int width, int endCol = 0) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(1, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(1, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);

  if (!endCol) {
    cpuB->colMax(*cpuA);
    gpuB->colMax(*gpuA);
  } else {
    MatrixPtr subCpuA = cpuA->subColMatrix(0, endCol);
    MatrixPtr subGpuA = gpuA->subColMatrix(0, endCol);
    MatrixPtr subCpuB = cpuB->subColMatrix(0, endCol);
    MatrixPtr subGpuB = gpuB->subColMatrix(0, endCol);
    subCpuB->colMax(*subCpuA);
    subGpuB->colMax(*subGpuA);
  }

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(1, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixCollectBias(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(1, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(1, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);

  real scale = 1.0f / (rand() % 10);  // NOLINT

  cpuA->collectBias(*cpuB, scale);
  gpuA->collectBias(*gpuB, scale);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(1, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixSumOfSquares(int height, int width, int endCol = 0) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, 1);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, 1);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuC = std::make_shared<GpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);

  if (!endCol) {
    cpuA->sumOfSquares(*cpuB, *cpuC);
    gpuA->sumOfSquares(*gpuB, *gpuC);
  } else {
    MatrixPtr subCpuB = cpuB->subColMatrix(0, endCol);
    MatrixPtr subCpuC = cpuC->subColMatrix(0, endCol);
    MatrixPtr subGpuB = gpuB->subColMatrix(0, endCol);
    MatrixPtr subGpuC = gpuC->subColMatrix(0, endCol);
    cpuA->sumOfSquares(*subCpuB, *subCpuC);
    gpuA->sumOfSquares(*subGpuB, *subGpuC);
  }

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, 1);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
}

void testMatrixBinaryClassificationError(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr gpuC = std::make_shared<GpuMatrix>(height, width);

  MatrixPtr cpuA2 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB2 = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuC2 = std::make_shared<CpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);
  cpuA2->copyFrom(*cpuA);
  cpuB2->copyFrom(*cpuB);
  cpuC2->copyFrom(*cpuC);

  real scale = 0.5;
  int columnOffset = rand() % width;  // NOLINT

  cpuA->binaryClassificationError(columnOffset, *cpuB, *cpuC, scale);
  gpuA->binaryClassificationError(columnOffset, *gpuB, *gpuC, scale);
  cpuA2->binaryClassificationError2(columnOffset, *cpuB2, *cpuC2, scale);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width);
  outputCheck->copyFrom(*gpuA);
  MatrixCheckErr(*cpuA, *outputCheck);
  MatrixCheckErr(*cpuA, *cpuA2);
}

TEST(Matrix, aggregate) {
  for (auto height : {1, 11, 16, 32, 64, 73, 128, 200, 1024, 2345}) {
    for (auto width : {1, 9, 16, 32, 64, 100, 512, 1000, 1024, 2453}) {
      VLOG(3) << " height=" << height << " width=" << width;
      testMatrixRowSum(height, width);
      testMatrixRowMax(height, width);
      testMatrixColSum(height, width);
      testMatrixColMax(height, width);
      testMatrixCollectBias(height, width);
      testMatrixTernaryRowDotMul(height, width);
      testMatrixAddDotMulVMM(height, width);

      testMatrixSumOfSquares(height, width);
      testMatrixBinaryClassificationError(height, width);
    }
  }
}

TEST(Matrix, aggregate2) {
  for (auto height : {16, 32, 128, 512, 1024}) {
    for (auto width :
         {16, 32, 64, 128, 256, 512, 768, 1024, 2048, 3072, 4096}) {
      VLOG(3) << " height=" << height << " width=" << width;

      int endCol = rand() % width;  // NOLINT
      testMatrixRowMax(height, width, endCol);
      testMatrixSumOfSquares(height, width, endCol);
      testMatrixColSum(height, width, endCol);
      testMatrixColMax(height, width, endCol);
      testMatrixAddDotMulVMM(height, width, endCol);
    }
  }
}

void testMatrixAddAtOffset(int height, int width1, int width2) {
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(height, width1);
  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(height, width2);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(height, width1);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(height, width2);

  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);
  cpuOutput->randomizeUniform();
  gpuOutput->copyFrom(*cpuOutput);

  int columnOffset = 0;
  int offset = std::abs(width1 - width2);
  if (offset) {
    columnOffset = rand() % offset;  // NOLINT
  }
  cpuOutput->addAtOffset(*cpuInput, columnOffset);
  gpuOutput->addAtOffset(*gpuInput, columnOffset);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width2);
  outputCheck->copyFrom(*gpuOutput);
  MatrixCheckEqual(*cpuOutput, *outputCheck);
}

void testMatrixAssignAtOffset(int height, int width1, int width2) {
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(height, width1);
  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(height, width2);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(height, width1);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(height, width2);

  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);
  cpuOutput->randomizeUniform();
  gpuOutput->copyFrom(*cpuOutput);

  int columnOffset = 0;
  int offset = std::abs(width1 - width2);
  if (offset) {
    columnOffset = rand() % offset;  // NOLINT
  }
  cpuOutput->assignAtOffset(*cpuInput, columnOffset);
  gpuOutput->assignAtOffset(*gpuInput, columnOffset);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, width2);
  outputCheck->copyFrom(*gpuOutput);
  MatrixCheckEqual(*cpuOutput, *outputCheck);
}

TEST(Matrix, AtOffset) {
  for (auto height : {1, 11, 73, 128, 200}) {
    for (auto width1 : {1, 32, 100, 512, 1000}) {
      for (auto width2 : {1, 32, 100, 512, 1000}) {
        VLOG(3) << " height=" << height << " width1=" << width1
                  << " width2=" << width2;

        testMatrixAddAtOffset(height, width1, width2);
        testMatrixAssignAtOffset(height, width1, width2);
      }
    }
  }
}

void testMatrixSelectRows(int numSamples, int tableSize, int inputDim) {
  MatrixPtr cpuTable = std::make_shared<CpuMatrix>(tableSize, inputDim);
  MatrixPtr gpuTable = std::make_shared<GpuMatrix>(tableSize, inputDim);
  cpuTable->randomizeUniform();
  gpuTable->copyFrom(*cpuTable);

  IVectorPtr cpuIds;
  IVectorPtr gpuIds;
  cpuIds = VectorT<int>::create(numSamples, false);
  gpuIds = VectorT<int>::create(numSamples, true);
  cpuIds->rand(tableSize);
  gpuIds->copyFrom(*cpuIds);

  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(numSamples, inputDim);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(numSamples, inputDim);
  cpuOutput->randomizeUniform();
  gpuOutput->copyFrom(*cpuOutput);

  cpuOutput->selectRows(*cpuTable, *cpuIds);
  gpuOutput->selectRows(*gpuTable, *gpuIds);

  // check
  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(numSamples, inputDim);
  outputCheck->copyFrom(*gpuOutput);
  MatrixCheckEqual(*cpuOutput, *outputCheck);
}

void testMatrixAddToRows(int numSamples, int tableSize, int inputDim) {
  MatrixPtr cpuTable = std::make_shared<CpuMatrix>(tableSize, inputDim);
  MatrixPtr gpuTable = std::make_shared<GpuMatrix>(tableSize, inputDim);
  cpuTable->randomizeUniform();
  gpuTable->copyFrom(*cpuTable);

  IVectorPtr cpuIds;
  IVectorPtr gpuIds;
  cpuIds = VectorT<int>::create(numSamples, false);
  gpuIds = VectorT<int>::create(numSamples, true);
  cpuIds->rand(tableSize);
  gpuIds->copyFrom(*cpuIds);

  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(numSamples, inputDim);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(numSamples, inputDim);
  cpuOutput->randomizeUniform();
  gpuOutput->copyFrom(*cpuOutput);

  cpuOutput->addToRows(*cpuTable, *cpuIds);
  gpuOutput->addToRows(*gpuTable, *gpuIds);

  // check
  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(tableSize, inputDim);
  outputCheck->copyFrom(*gpuTable);
  MatrixCheckErr(*cpuTable, *outputCheck);
}

TEST(Matrix, tableProjection) {
  for (auto numSamples : {10, 100, 1000, 10000, 80000}) {
    for (auto tableSize : {10, 100}) {
      for (auto inputDim : {20, 50}) {
        VLOG(3) << " numSamples=" << numSamples << " tableSize=" << tableSize
                  << " inputDim=" << inputDim;
        testMatrixSelectRows(numSamples, tableSize, inputDim);
        testMatrixAddToRows(numSamples, tableSize, inputDim);
      }
    }
  }
}

void testMatrixMul(bool transa, bool transb, int dimM, int dimN, int dimK) {
  int heightA = transa == false ? dimM : dimK;
  int widthA = transa == false ? dimK : dimM;
  int heightB = transb == false ? dimK : dimN;
  int widthB = transb == false ? dimN : dimK;
  int heightC = dimM;
  int widthC = dimN;

  MatrixPtr cpuA = std::make_shared<CpuMatrix>(heightA, widthA, transa);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(heightB, widthB, transb);
  MatrixPtr cpuC = std::make_shared<CpuMatrix>(heightC, widthC);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(heightA, widthA, transa);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(heightB, widthB, transb);
  MatrixPtr gpuC = std::make_shared<GpuMatrix>(heightC, widthC);

  real alpha = 1.5;
  real beta = 2.0;
  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);

  cpuC->mul(cpuA, cpuB, alpha, beta);
  gpuC->mul(gpuA, gpuB, alpha, beta);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(heightC, widthC);
  outputCheck->copyFrom(*gpuC);
  MatrixCheckErr(*cpuC, *outputCheck);
}

void testSubMatrixMul(bool transa, bool transb, int dimM, int dimN, int dimK) {
  int heightA = transa == false ? dimM : dimK;
  int widthA = transa == false ? dimK : dimM;
  int heightB = transb == false ? dimK : dimN;
  int widthB = transb == false ? dimN : dimK;
  int heightC = dimM;
  int widthC = dimN;

  MatrixPtr cpuA = std::make_shared<CpuMatrix>(heightA, widthA, transa);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(heightB, widthB, transb);
  MatrixPtr cpuC = std::make_shared<CpuMatrix>(heightC, widthC);
  MatrixPtr gpuA = std::make_shared<GpuMatrix>(heightA, widthA, transa);
  MatrixPtr gpuB = std::make_shared<GpuMatrix>(heightB, widthB, transb);
  MatrixPtr gpuC = std::make_shared<GpuMatrix>(heightC, widthC);

  real alpha = 1.5;
  real beta = 2.0;
  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuC->randomizeUniform();
  gpuA->copyFrom(*cpuA);
  gpuB->copyFrom(*cpuB);
  gpuC->copyFrom(*cpuC);

  auto subSize = [](int& start, int& end, int dim) {
    if (dim == 1) {
      start = 0;
      end = dim;
    } else {
      int subDim = rand() % (dim - 1) + 1;  // NOLINT
      start = rand() % (dim - subDim);      // NOLINT
      end = start + subDim;
    }
  };

  auto subMatrix = [](MatrixPtr& sub, MatrixPtr matrix, size_t startRow,
                      size_t endRow, size_t startCol, size_t endCol) {
    if (!matrix->isTransposed()) {
      sub = matrix->subMatrix(startRow, endRow, startCol, endCol);
    } else {
      sub = matrix->subMatrix(startCol, endCol, startRow, endRow);
    }
  };

  int startM, endM;
  int startN, endN;
  int startK, endK;
  subSize(startM, endM, dimM);
  subSize(startN, endN, dimN);
  subSize(startK, endK, dimK);

  MatrixPtr subCpuA;
  MatrixPtr subCpuB;
  MatrixPtr subGpuA;
  MatrixPtr subGpuB;
  subMatrix(subCpuA, cpuA, startM, endM, startK, endK);
  subMatrix(subGpuA, gpuA, startM, endM, startK, endK);
  subMatrix(subCpuB, cpuB, startK, endK, startN, endN);
  subMatrix(subGpuB, gpuB, startK, endK, startN, endN);
  MatrixPtr subCpuC = cpuC->subMatrix(startM, endM, startN, endN);
  MatrixPtr subGpuC = gpuC->subMatrix(startM, endM, startN, endN);

  subCpuC->mul(subCpuA, subCpuB, alpha, beta);
  subGpuC->mul(subGpuA, subGpuB, alpha, beta);

  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(heightC, widthC);
  outputCheck->copyFrom(*gpuC);
  MatrixCheckErr(*cpuC, *outputCheck);
}

TEST(Matrix, mul) {
  for (auto transa : {false, true}) {
    for (auto transb : {false, true}) {
      for (auto dimM : {1, 9, 53, 127, 345, 1023, 2135}) {
        for (auto dimN : {1, 5, 37, 256, 1024}) {
          for (auto dimK : {8, 45, 346, 784, 1025}) {
            if (true == transa && true == transb) {
              continue;
            }
            VLOG(3) << setiosflags(ios::left) << setfill(' ')
                      << " transa=" << transa << " transb=" << transb
                      << " dimM=" << setw(5) << dimM << " dimN=" << setw(5)
                      << dimN << " dimK=" << setw(5) << dimK;

            testMatrixMul(transa, transb, dimM, dimN, dimK);
            testSubMatrixMul(transa, transb, dimM, dimN, dimK);
          }
        }
      }
    }
  }
}

void testVectorRowFunc(int size) {
  CpuVectorPtr cpu = std::make_shared<CpuVectorT<real>>(size);
  GpuVectorPtr gpu = std::make_shared<GpuVectorT<real>>(size);

  cpu->rand();
  gpu->copyFrom(*cpu);

  EXPECT_EQ(cpu->getMax(), gpu->getMax());
  EXPECT_EQ(cpu->getMin(), gpu->getMin());
  EXPECT_EQ(cpu->getAbsMax(), gpu->getAbsMax());
}

TEST(Vector, rowFunc) {
  for (auto size : {1, 5, 31, 90, 150, 500, 1000, 4000}) {
    VLOG(3) << " size=" << size;
    testVectorRowFunc(size);
  }
}

template<class T>
void testVectorReset(int size) {
  std::shared_ptr<CpuVectorT<T>> cpu = std::make_shared<CpuVectorT<T>>(size);
  std::shared_ptr<GpuVectorT<T>> gpu = std::make_shared<GpuVectorT<T>>(size);

  T value = (T)((int)rand() % 100 + 1.0f / ((int)rand() % 100));
  cpu->reset(value);
  gpu->reset(value);

  std::shared_ptr<CpuVectorT<T>> out = std::make_shared<CpuVectorT<T>>(size);
  out->copyFrom(*gpu);
  VectorCheckEqual(*cpu, *out);
}

template<class T>
void testVecortSelectFrom(int size) {
  std::shared_ptr<CpuVectorT<T>> cpuDst = std::make_shared<CpuVectorT<T>>(size);
  std::shared_ptr<GpuVectorT<T>> gpuDst = std::make_shared<GpuVectorT<T>>(size);
  std::shared_ptr<CpuVectorT<T>>
    cpuSrc = std::make_shared<CpuVectorT<T>>(size*2);
  std::shared_ptr<GpuVectorT<T>>
    gpuSrc = std::make_shared<GpuVectorT<T>>(size*2);
  CpuIVectorPtr cpuIds = std::make_shared<CpuVectorT<int>>(size);
  GpuIVectorPtr gpuIds = std::make_shared<GpuVectorT<int>>(size);

  if (std::is_same<T, real>::value) {
    cpuSrc->rand();
  } else {
    cpuSrc->rand(100000);
  }
  gpuSrc->copyFrom(*cpuSrc);
  cpuIds->rand(size);
  gpuIds->copyFrom(*cpuIds);

  cpuDst->selectFrom(*cpuSrc, *cpuIds);
  gpuDst->selectFrom(*gpuSrc, *gpuIds);

  std::shared_ptr<CpuVectorT<T>> out = std::make_shared<CpuVectorT<T>>(size);
  out->copyFrom(*gpuDst);
  VectorCheckEqual(*cpuDst, *out);
}

template<class T>
void testVecotrZeroMem(int size) {
  std::shared_ptr<CpuVectorT<T>> cpu = std::make_shared<CpuVectorT<T>>(size);
  std::shared_ptr<GpuVectorT<T>> gpu = std::make_shared<GpuVectorT<T>>(size);

  cpu->zeroMem();
  gpu->zeroMem();

  std::shared_ptr<CpuVectorT<T>> out = std::make_shared<CpuVectorT<T>>(size);
  out->copyFrom(*gpu);
  VectorCheckEqual(*cpu, *out);
}

template<class T>
void testVectorIsEqual(int size) {
  std::shared_ptr<CpuVectorT<T>> cpuA = std::make_shared<CpuVectorT<T>>(size);
  std::shared_ptr<CpuVectorT<T>> cpuB = std::make_shared<CpuVectorT<T>>(size);
  std::shared_ptr<GpuVectorT<T>> gpuA = std::make_shared<GpuVectorT<T>>(size);
  std::shared_ptr<GpuVectorT<T>> gpuB = std::make_shared<GpuVectorT<T>>(size);

  if (std::is_same<T, real>::value) {
    cpuB->rand();
  } else {
    cpuB->rand(100000);
  }
  gpuB->copyFrom(*cpuB);

  T value = (T)((int)rand() % 100 + 1.0f / ((int)rand() % 100));
  cpuA->isEqualTo(*cpuB, value);
  gpuA->isEqualTo(*gpuB, value);

  std::shared_ptr<CpuVectorT<T>> out = std::make_shared<CpuVectorT<T>>(size);
  out->copyFrom(*gpuA);
  VectorCheckEqual(*cpuA, *out);
}

TEST(Vector, Equal) {
  for (auto size : {1, 5, 31, 90, 150, 500, 1000, 4000}) {
    VLOG(3) << " size=" << size;
    testVectorReset<int>(size);
    testVectorReset<real>(size);
    testVecortSelectFrom<int>(size);
    testVecortSelectFrom<real>(size);
    testVecotrZeroMem<int>(size);
    testVecotrZeroMem<real>(size);
    testVectorIsEqual<int>(size);
    testVectorIsEqual<real>(size);
  }
}

void testMatrixTopK(int samples, int dim, int beamSize) {
  MatrixPtr cpuSrc = std::make_shared<CpuMatrix>(samples, dim);
  MatrixPtr gpuSrc = std::make_shared<GpuMatrix>(samples, dim);
  MatrixPtr cpuVal = std::make_shared<CpuMatrix>(samples, beamSize);
  MatrixPtr gpuVal = std::make_shared<GpuMatrix>(samples, beamSize);
  IVectorPtr cpuIds = std::make_shared<CpuIVector>(samples * beamSize);
  IVectorPtr gpuIds = std::make_shared<GpuIVector>(samples * beamSize);

  cpuSrc->randomizeUniform();
  gpuSrc->copyFrom(*cpuSrc);

  cpuSrc->rowMax(*cpuIds, *cpuVal);
  gpuSrc->rowMax(*gpuIds, *gpuVal);

  MatrixPtr outVal = std::make_shared<CpuMatrix>(samples, beamSize);
  outVal->copyFrom(*gpuVal);
  MatrixCheckEqual(*cpuVal, *outVal);
}

TEST(Matrix, topK) {
  for (auto samples : {1, 5, 31, 90, 150, 500}) {
    for (auto dim : {1, 5 , 8, 10, 15, 64, 80, 120, 256, 300,
                     1280, 5120, 50000}) {
      for (auto beamSize : {1, 5, 10, 20, 40, (int)rand() % dim + 1}) {
        if (beamSize > dim) continue;
        VLOG(3) << " samples=" << samples
                << " beamSize=" << beamSize
                << " dim=" << dim;
        testMatrixTopK(samples, dim, beamSize);
      }
    }
  }
}

void testSMatrixTopK(int samples, int dim, int beamSize, real ratio) {
  int nnz = samples * dim * ratio;
  MatrixPtr cpuSrc = std::make_shared<CpuSparseMatrix>(samples, dim, nnz);
  MatrixPtr gpuSrc = std::make_shared<GpuSparseMatrix>(samples, dim, nnz);
  MatrixPtr cpuVal = std::make_shared<CpuMatrix>(samples, beamSize);
  MatrixPtr gpuVal = std::make_shared<GpuMatrix>(samples, beamSize);
  IVectorPtr cpuIds = std::make_shared<CpuIVector>(samples * beamSize);
  IVectorPtr gpuIds = std::make_shared<GpuIVector>(samples * beamSize);

  cpuSrc->randomizeUniform();
  gpuSrc->copyFrom(*cpuSrc);
  cpuVal->zero();
  cpuIds->zero();
  gpuVal->zero();
  gpuIds->zero();

  cpuSrc->rowMax(*cpuIds, *cpuVal);
  gpuSrc->rowMax(*gpuIds, *gpuVal);

  MatrixPtr outCheckMaxVal = std::make_shared<CpuMatrix>(samples, beamSize);
  outCheckMaxVal->copyFrom(*gpuVal);
  MatrixCheckEqual(*cpuVal, *outCheckMaxVal);

  IVectorPtr outCheckIds = std::make_shared<CpuIVector>(samples * beamSize);
  outCheckIds->copyFrom(*gpuIds);

  const int* data1 = cpuIds->getData();
  const int* data2 = outCheckIds->getData();
  size_t size = cpuIds->getSize();
  for (size_t i = 0; i < size; i++) {
    if (data1[i] == -1 && data1[i] != data2[i]) {
      EXPECT_EQ(data1[i], data2[i]);
    }
  }
}

TEST(SMatrix, topK) {
  for (auto samples : {1, 5, 100}) {
    for (auto dim : {10000, 10000, 50000}) {
      for (auto beamSize : {1, 5, 40, 100, 500}) {
        for (auto ratio : {0.01, 0.001}) {
          if (beamSize > dim) continue;
          VLOG(3) << " samples=" << samples
                  << " beamSize=" << beamSize
                  << " dim=" << dim
                  << " ratio=" << ratio;
          testSMatrixTopK(samples, dim, beamSize, ratio);
        }
      }
    }
  }
}

void testMatrixCopyByRowIndex(int outHeight, int inHeight, int width) {
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(inHeight, width);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(inHeight, width);
  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);

  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(outHeight, width);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(outHeight, width);
  cpuOutput->zero();
  gpuOutput->zero();

  IVectorPtr cpuRowIndex = IVector::create(outHeight, false);
  IVectorPtr gpuRowIndex = IVector::create(outHeight, true);
  cpuRowIndex->rand(inHeight);
  gpuRowIndex->copyFrom(*cpuRowIndex);

  cpuOutput->copyByRowIndex(*cpuInput, *cpuRowIndex);
  gpuOutput->copyByRowIndex(*gpuInput, *gpuRowIndex);

  // check
  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(outHeight, width);
  outputCheck->copyFrom(*gpuOutput);
  MatrixCheckEqual(*cpuOutput, *outputCheck);
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

void testMatrixSequenceAvgForward(int batchSize, int inputDim, int mode) {
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(batchSize, inputDim);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(batchSize, inputDim);
  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);

  IVectorPtr cpuSequence;
  generateSequenceStartPositions(batchSize, cpuSequence);
  IVectorPtr gpuSequence = IVector::create(cpuSequence->getSize(), true);
  gpuSequence->copyFrom(*cpuSequence);

  int newBatchSize = cpuSequence->getSize() - 1;
  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(newBatchSize, inputDim);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(newBatchSize, inputDim);
  cpuOutput->zero();
  gpuOutput->zero();

  cpuOutput->sequenceAvgForward(*cpuInput, *cpuSequence, mode);
  gpuOutput->sequenceAvgForward(*gpuInput, *gpuSequence, mode);

  // check
  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(newBatchSize, inputDim);
  outputCheck->copyFrom(*gpuOutput);
  MatrixCheckErr(*cpuOutput, *outputCheck);
}

TEST(Matrix, sequenceAvgForward) {
  for (auto batchSize : {10, 128, 6000}) {
    for (auto inputDim : {32, 100, 512}) {
      for (auto mode : {0, 1, 2}) {
        VLOG(3) << " batchSize=" << batchSize << " inputDim=" << inputDim
                << " mode=" << mode;
        testMatrixSequenceAvgForward(batchSize, inputDim, mode);
      }
    }
  }
}

void testCosSim(int heightX, int heightY, int width, real scale) {
  MatrixPtr prevOutX = CpuMatrix::create(heightX, width, false, false);
  MatrixPtr prevOutY = CpuMatrix::create(heightY, width, false, false);
  MatrixPtr output = CpuMatrix::create(heightX, 1, false, false);

  prevOutX->randomizeUniform();
  prevOutY->randomizeUniform();
  prevOutX->add(-0.5);
  prevOutY->add(-0.5);
  output->randomizeUniform();

  MatrixPtr prevOutXGpu = GpuMatrix::create(heightX, width, false, true);
  MatrixPtr prevOutYGpu = GpuMatrix::create(heightY, width, false, true);
  MatrixPtr outputGpu = GpuMatrix::create(heightX, 1, false, true);

  prevOutXGpu->copyFrom(*prevOutX);
  prevOutYGpu->copyFrom(*prevOutY);
  outputGpu->copyFrom(*output);

  output->cosSim(*prevOutX, *prevOutY, scale);
  outputGpu->cosSim(*prevOutXGpu, *prevOutYGpu, scale);

  MatrixPtr outputCheck = CpuMatrix::create(heightX, 1, false, false);
  outputCheck->copyFrom(*outputGpu);
  MatrixCheckErr(*output, *outputCheck);
}

TEST(Matrix, cosSim) {
  for (auto heightX : {10, 100, 1000}) {
    for (auto heightY : {1, heightX}) {
      for (auto width : {10, 100, 1000}) {
        for (auto scale : {1.0, 2.0}) {
          testCosSim(heightX, heightY, width, scale);
        }
      }
    }
  }
}

void testCosSimDerivate(int heightX, int heightY, int width,
                        real scale) {
  MatrixPtr prevOutX = CpuMatrix::create(heightX, width, false, false);
  MatrixPtr prevOutY = CpuMatrix::create(heightY, width, false, false);
  MatrixPtr grad = CpuMatrix::create(heightX, 1, false, false);
  MatrixPtr output = CpuMatrix::create(heightX, 1, false, false);
  MatrixPtr prevGradX = CpuMatrix::create(heightX, width, false, false);
  MatrixPtr prevGradY = CpuMatrix::create(heightY, width, false, false);

  prevOutX->randomizeUniform();
  prevOutY->randomizeUniform();
  grad->randomizeUniform();
  output->randomizeUniform();
  prevGradX->randomizeUniform();
  prevGradY->randomizeUniform();

  MatrixPtr prevOutXGpu = GpuMatrix::create(heightX, width, false, true);
  MatrixPtr prevOutYGpu = GpuMatrix::create(heightY, width, false, true);
  MatrixPtr gradGpu = GpuMatrix::create(heightX, 1, false, true);
  MatrixPtr outputGpu = GpuMatrix::create(heightX, 1, false, true);
  MatrixPtr prevGradXGpu = GpuMatrix::create(heightX, width, false, true);
  MatrixPtr prevGradYGpu = GpuMatrix::create(heightY, width, false, true);

  prevOutXGpu->copyFrom(*prevOutX);
  prevOutYGpu->copyFrom(*prevOutY);
  gradGpu->copyFrom(*grad);
  outputGpu->copyFrom(*output);
  prevGradXGpu->copyFrom(*prevGradX);
  prevGradYGpu->copyFrom(*prevGradY);

  grad->cosSimDerivative(*output,
                         *prevOutX,
                         *prevOutY,
                         *prevGradX,
                         *prevGradY,
                         scale);

  gradGpu->cosSimDerivative(*outputGpu,
                            *prevOutXGpu,
                            *prevOutYGpu,
                            *prevGradXGpu,
                            *prevGradYGpu,
                            scale);

  MatrixPtr prevGradXCheck = CpuMatrix::create(heightX, width, false,
                                               false);
  MatrixPtr prevGradYCheck = CpuMatrix::create(heightY, width, false,
                                               false);
  prevGradXCheck->copyFrom(*prevGradXGpu);
  prevGradYCheck->copyFrom(*prevGradYGpu);
  MatrixCheckErr(*prevGradX, *prevGradXCheck);
  MatrixCheckErr(*prevGradY, *prevGradYCheck);
}

TEST(Matrix, cosSimDerivate) {
  for (auto heightX : {1, 10, 100}) {
    for (auto heightY : {1, heightX}) {
      for (auto width : {1, 10, 100}) {
        for (auto scale : {1.0, 2.0}) {
          testCosSimDerivate(heightX, heightY, width, scale);
        }
      }
    }
  }
}

void testParamReluForward(int height, int width, int w_height,
                                                 int w_width) {
  MatrixPtr output = CpuMatrix::create(height, width, false, false);
  MatrixPtr input = CpuMatrix::create(height, width, false, false);
  MatrixPtr w = CpuMatrix::create(w_height, w_width, false, false);

  output->randomizeUniform();
  input->randomizeUniform();
  w->randomizeUniform();
  input->add(-0.5);

  MatrixPtr outputGpu = GpuMatrix::create(height, width, false, true);
  MatrixPtr inputGpu = GpuMatrix::create(height, width, false, true);
  MatrixPtr wGpu = GpuMatrix::create(w_height, w_width, false, true);

  inputGpu->copyFrom(*input);
  wGpu->copyFrom(*w);

  output->paramReluForward(*input, *w);
  outputGpu->paramReluForward(*inputGpu, *wGpu);

  MatrixPtr outputCheck = CpuMatrix::create(height, width, false, false);
  outputCheck->copyFrom(*outputGpu);
  MatrixCheckEqual(*output, *outputCheck);
}

TEST(Matrix, paramReluForward) {
  for (auto height : {10, 100}) {
    for (auto width : {10, 100}) {
      for (auto w_height : {1, 2}) {
        for (auto w_width : {1, 2}) {
          testParamReluForward(height, width, w_height, w_width);
        }
      }
    }
  }
}

void testParamReluBackwardW(int height, int width, int w_height,
                                                   int w_width) {
  MatrixPtr oGrad = CpuMatrix::create(height, width, false, false);
  MatrixPtr input = CpuMatrix::create(height, width, false, false);
  MatrixPtr w = CpuMatrix::create(w_height, w_width, false, false);

  oGrad->randomizeUniform();
  input->randomizeUniform();
  w->randomizeUniform();
  input->add(-0.5);

  MatrixPtr oGradGpu = GpuMatrix::create(height, width, false, true);
  MatrixPtr inputGpu = GpuMatrix::create(height, width, false, true);
  MatrixPtr wGpu = GpuMatrix::create(w_height, w_width, false, true);

  oGradGpu->copyFrom(*oGrad);
  inputGpu->copyFrom(*input);
  wGpu->copyFrom(*w);

  w->paramReluBackwardW(*oGrad, *input);
  wGpu->paramReluBackwardW(*oGradGpu, *inputGpu);
  MatrixPtr wCheck = CpuMatrix::create(w_height, w_width, false, false);
  wCheck->copyFrom(*wGpu);
  MatrixCheckErr(*w, *wCheck);
}

TEST(Matrix, paramReluBackwardW) {
  for (auto height : {10, 100}) {
    for (auto width : {10, 100}) {
      for (auto w_height : {1, 2}) {
        for (auto w_width : {1, 2}) {
          testParamReluBackwardW(height, width, w_height, w_width);
        }
      }
    }
  }
}

void testParamReluBackwardDiff(int height, int width, int w_height,
                                                      int w_width) {
  MatrixPtr oGrad = CpuMatrix::create(height, width, false, false);
  MatrixPtr input = CpuMatrix::create(height, width, false, false);
  MatrixPtr diff = CpuMatrix::create(height, width, false, false);
  MatrixPtr w = CpuMatrix::create(w_height, w_width, false, false);

  oGrad->randomizeUniform();
  input->randomizeUniform();
  w->randomizeUniform();
  diff->randomizeUniform();
  input->add(-0.5);

  MatrixPtr oGradGpu = GpuMatrix::create(height, width, false, true);
  MatrixPtr inputGpu = GpuMatrix::create(height, width, false, true);
  MatrixPtr diffGpu = CpuMatrix::create(height, width, false, true);
  MatrixPtr wGpu = GpuMatrix::create(w_height, w_width, false, true);

  oGradGpu->copyFrom(*oGrad);
  inputGpu->copyFrom(*input);
  wGpu->copyFrom(*w);
  diffGpu->copyFrom(*diff);

  diff->paramReluBackwardDiff(*oGrad, *input, *w);
  diffGpu->paramReluBackwardDiff(*oGradGpu, *inputGpu, *wGpu);

  MatrixPtr diffCheck = CpuMatrix::create(height, width, false, false);
  diffCheck->copyFrom(*diffGpu);
  MatrixCheckErr(*diff, *diffCheck);
}

TEST(Matrix, paramReluBackwardDiff) {
  for (auto height : {10, 100}) {
    for (auto width : {10, 100}) {
      for (auto w_height : {1, 2}) {
        for (auto w_width : {1, 2}) {
          testParamReluBackwardDiff(height, width, w_height, w_width);
        }
      }
    }
  }
}

void testClassificationError(int numSamples, int dim) {
  MatrixPtr cpuError = std::make_shared<CpuMatrix>(numSamples, 1);
  MatrixPtr gpuError = std::make_shared<GpuMatrix>(numSamples, 1);
  MatrixPtr cpuOutput = std::make_shared<CpuMatrix>(numSamples, dim);
  MatrixPtr gpuOutput = std::make_shared<GpuMatrix>(numSamples, dim);
  IVectorPtr cpuLabel = std::make_shared<CpuIVector>(numSamples);
  IVectorPtr gpuLabel = std::make_shared<GpuIVector>(numSamples);

  cpuOutput->randomizeUniform();
  cpuLabel->rand(dim);
  gpuOutput->copyFrom(*cpuOutput);
  gpuLabel->copyFrom(*cpuLabel);

  cpuError->classificationError(cpuOutput, cpuLabel);
  gpuError->classificationError(gpuOutput, gpuLabel);

  MatrixPtr check = std::make_shared<CpuMatrix>(numSamples, 1);
  check->copyFrom(*gpuError);
  MatrixCheckEqual(*cpuError, *check);
}

TEST(Matrix, classificationError) {
  for (auto numSamples : {1, 10, 100, 1000, 70000}) {
    for (auto dim : {1, 10, 100, 1000}) {
      VLOG(3) << " numSamples=" << numSamples << " dim=" << dim;
      testClassificationError(numSamples, dim);
    }
  }
}

void testMaxPoolFwdBwd(int numSamples, int channels,
                       int imgSizeH, int imgSizeW,
                       int ksizeH, int ksizeW,
                       int strideH, int strideW,
                       int padH, int padW) {
  int outH = 0, outW = 0;
  outH = (imgSizeH - ksizeH + 2 * padH + strideH - 1) / strideH + 1;
  outW = (imgSizeW - ksizeW + 2 * padW + strideW - 1) / strideW + 1;

  int inWidth = imgSizeH * imgSizeW * channels;
  MatrixPtr input = CpuMatrix::create(numSamples, inWidth, false, false);
  MatrixPtr inputGpu = GpuMatrix::create(numSamples, inWidth, false, true);

  int outWidth = channels * outH * outW;
  MatrixPtr target = CpuMatrix::create(numSamples, outWidth, false, false);
  MatrixPtr targetGpu = GpuMatrix::create(numSamples, outWidth, false, true);

  input->randomizeUniform();
  target->randomizeUniform();
  inputGpu->copyFrom(*input);
  targetGpu->copyFrom(*target);

  target->maxPoolForward(*input, imgSizeH, imgSizeW,
                         channels, ksizeW, ksizeH,
                         strideH, strideW, outH, outW, padH, padW);
  targetGpu->maxPoolForward(*inputGpu, imgSizeH, imgSizeW,
                            channels, ksizeW, ksizeH,
                            strideH, strideW, outH, outW, padH, padW);
  MatrixPtr targetCheck = CpuMatrix::create(numSamples, outWidth, false, false);
  targetCheck->copyFrom(*targetGpu);
  checkMatrixEqual(target, targetCheck);

  MatrixPtr inputGrad = CpuMatrix::create(numSamples, inWidth, false, false);
  MatrixPtr inputGpuGrad = GpuMatrix::create(numSamples, inWidth, false, true);
  MatrixPtr targetGrad = CpuMatrix::create(numSamples, outWidth, false, false);
  MatrixPtr targetGpuGrad = GpuMatrix::create(numSamples, outWidth,
                                              false, true);

  inputGrad->randomizeUniform();
  targetGrad->randomizeUniform();
  inputGpuGrad->copyFrom(*inputGrad);
  targetGpuGrad->copyFrom(*targetGrad);

  inputGrad->maxPoolBackward(*input, imgSizeH, imgSizeW,
                             *targetGrad, *target,
                             ksizeW, ksizeH,
                             strideH, strideW,
                             outH, outW, 1.0, 1.0, padH, padW);
  inputGpuGrad->maxPoolBackward(*inputGpu, imgSizeH, imgSizeW,
                                *targetGpuGrad, *targetGpu,
                                ksizeW, ksizeH,
                                strideH, strideW,
                                outH, outW, 1.0, 1.0, padH, padW);
  MatrixPtr targetBwdCheck = CpuMatrix::create(numSamples, inWidth,
                                               false, false);
  targetBwdCheck->copyFrom(*inputGpuGrad);
  checkMatrixEqual(inputGrad, targetBwdCheck);
}

void testAvgPoolFwdBwd(int numSamples, int channels,
                       int imgSizeH, int imgSizeW,
                       int ksizeH, int ksizeW,
                       int strideH, int strideW,
                       int padH, int padW) {
  int outH = 0, outW = 0;
  outH = (imgSizeH - ksizeH + 2 * padH + strideH - 1) / strideH + 1;
  outW = (imgSizeW - ksizeW + 2 * padW + strideW - 1) / strideW + 1;

  int inWidth = imgSizeH * imgSizeW * channels;
  MatrixPtr input = CpuMatrix::create(numSamples, inWidth, false, false);
  MatrixPtr inputGpu = GpuMatrix::create(numSamples, inWidth, false, true);

  int outWidth = channels * outH * outW;
  MatrixPtr target = CpuMatrix::create(numSamples, outWidth, false, false);
  MatrixPtr targetGpu = GpuMatrix::create(numSamples, outWidth, false, true);

  input->randomizeUniform();
  target->randomizeUniform();
  inputGpu->copyFrom(*input);
  targetGpu->copyFrom(*target);

  target->avgPoolForward(*input, imgSizeH, imgSizeW,
                         channels, ksizeW, ksizeH,
                         strideH, strideW, outH, outW, padH, padW);
  targetGpu->avgPoolForward(*inputGpu, imgSizeH, imgSizeW,
                            channels, ksizeW, ksizeH,
                            strideH, strideW, outH, outW, padH, padW);
  MatrixPtr targetCheck = CpuMatrix::create(numSamples, outWidth, false, false);
  targetCheck->copyFrom(*targetGpu);
  MatrixCheckErr(*target, *targetCheck);

  MatrixPtr inputGrad = CpuMatrix::create(numSamples, inWidth, false, false);
  MatrixPtr inputGpuGrad = GpuMatrix::create(numSamples, inWidth, false, true);
  MatrixPtr targetGrad = CpuMatrix::create(numSamples, outWidth, false, false);
  MatrixPtr targetGpuGrad = GpuMatrix::create(numSamples, outWidth,
                                              false, true);

  inputGrad->randomizeUniform();
  targetGrad->randomizeUniform();
  inputGpuGrad->copyFrom(*inputGrad);
  targetGpuGrad->copyFrom(*targetGrad);

  inputGrad->avgPoolBackward(*targetGrad, imgSizeH, imgSizeW,
                             ksizeW, ksizeH,
                             strideH, strideW,
                             outH, outW, 1.0, 1.0, padH, padW);
  inputGpuGrad->avgPoolBackward(*targetGpuGrad, imgSizeH, imgSizeW,
                                ksizeW, ksizeH,
                                strideH, strideW,
                                outH, outW, 1.0, 1.0, padH, padW);
  MatrixPtr targetBwdCheck = CpuMatrix::create(numSamples, inWidth,
                                               false, false);
  targetBwdCheck->copyFrom(*inputGpuGrad);
  MatrixCheckErr(*inputGrad, *targetBwdCheck);
}

TEST(Matrix, PoolFwdBwd) {
  for (auto numSamples : {5, 32}) {
    for (auto channels : {1, 9, 32}) {
      for (auto imgSizeH : {14, 28}) {
        for (auto imgSizeW : {16, 30}) {
          for (auto sizeX : {2, 5}) {
            for (auto sizeY : {2, 5}) {
              for (auto sH : {1, 2}) {
                for (auto sW : {1, 2}) {
                   for (auto pH : {0, (sizeY - 1)/2}) {
                     for (auto pW : {0, (sizeX - 1)/2}) {
                       VLOG(3) << " numSamples=" << numSamples
                               << " channels=" << channels
                               << " imgSizeH=" << imgSizeH
                               << " imgSizeW=" << imgSizeW
                               << " sizeX=" << sizeX
                               << " sizeY=" << sizeY
                               << " strideH=" << sH
                               << " strideW=" << sW
                               << " padingH=" << pH
                               << " padingW=" << pW;
                       testMaxPoolFwdBwd(numSamples, channels, imgSizeH,
                         imgSizeW, sizeX, sizeY, sH, sW, pH, pW);
                       testAvgPoolFwdBwd(numSamples, channels, imgSizeH,
                         imgSizeW, sizeX, sizeY, sH, sW, pH, pW);
                     }
                   }
                }
              }
            }
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  return RUN_ALL_TESTS();
}

#endif
