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
/// This unittest checks GpuMatrix/CpuMatrix get same result, so disable when
/// only cpu version.

#include <gtest/gtest.h>
#include "TensorCheck.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/SparseMatrix.h"
#include "paddle/testing/TestUtil.h"
#include "paddle/utils/DynamicLoader.h"
#include "paddle/utils/Stat.h"
#include "paddle/utils/Util.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT
using autotest::TensorCheckEqual;
using autotest::TensorCheckErr;

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

  TensorCheckEqual(*cpuOutput, *gpuOutput);
  TensorCheckEqual(*cpuIndex, *gpuIndex);

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

  TensorCheckEqual(*cpuInputGrad, *gpuInputGrad);
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

void testMatrixGetMinMax(int height, int width) {
  MatrixPtr cpuInput = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpuInput = std::make_shared<GpuMatrix>(height, width);
  cpuInput->randomizeUniform();
  gpuInput->copyFrom(*cpuInput);

  real cpuMin = cpuInput->getMin();
  real gpuMin = gpuInput->getMin();
  real cpuMax = cpuInput->getMax();
  real gpuMax = gpuInput->getMax();

  EXPECT_EQ(cpuMin, gpuMin);
  EXPECT_EQ(cpuMax, gpuMax);
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

  if (numColumns == 0) return;

  cpuA->zeroAtOffset(columnOffset, numColumns);
  gpuA->zeroAtOffset(columnOffset, numColumns);

  /* cpuTest */
  real* a = cpuTest->getData() + columnOffset;
  for (int64_t i = 0; i < height; ++i) {
    for (int64_t j = 0; j < numColumns; ++j) {
      a[i * width + j] = 0;
    }
  }

  TensorCheckEqual(*cpuA, *gpuA);
  TensorCheckEqual(*cpuA, *cpuTest);
}

void testMatrixDeepSwap(int height, int width) {
  MatrixPtr cpuA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuB = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuCopyA = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr cpuCopyB = std::make_shared<CpuMatrix>(height, width);

  cpuA->randomizeUniform();
  cpuB->randomizeUniform();
  cpuCopyA->copyFrom(*cpuA);
  cpuCopyB->copyFrom(*cpuB);

  // swap matrix cpuA and cpuB
  cpuA->deepSwap(*cpuB);

  TensorCheckEqual(*cpuA, *cpuCopyB);
  TensorCheckEqual(*cpuB, *cpuCopyA);
}

void testMatrixTranspose(int height, int width) {
  MatrixPtr cpu = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpu = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr cpuT = std::make_shared<CpuMatrix>(width, height);
  MatrixPtr gpuT = std::make_shared<GpuMatrix>(width, height);

  cpu->randomizeUniform();
  gpu->copyFrom(*cpu);
  cpu->transpose(cpuT, false);
  gpu->transpose(gpuT, true);

  TensorCheckEqual(*cpuT, *gpuT);
}

void testMatrixRotate(int height, int width) {
  MatrixPtr cpu = std::make_shared<CpuMatrix>(height, width);
  MatrixPtr gpu = std::make_shared<GpuMatrix>(height, width);
  MatrixPtr cpuR = std::make_shared<CpuMatrix>(width, height);
  MatrixPtr gpuR = std::make_shared<GpuMatrix>(width, height);

  cpu->randomizeUniform();
  gpu->copyFrom(*cpu);

  cpu->rotate(cpuR, false, true);
  gpu->rotate(gpuR, true, true);
  TensorCheckEqual(*cpuR, *gpuR);

  cpu->rotate(cpuR, true, false);
  gpu->rotate(gpuR, false, false);
  TensorCheckEqual(*cpuR, *gpuR);
}

void testMatrixInverse(int height) {
  MatrixPtr cpu = std::make_shared<CpuMatrix>(height, height);
  MatrixPtr gpu = std::make_shared<GpuMatrix>(height, height);
  MatrixPtr cpuI = std::make_shared<CpuMatrix>(height, height);
  MatrixPtr gpuI = std::make_shared<GpuMatrix>(height, height);

  /* Make matrix well conditioned: cpu * cpuT + Identity */
  cpu->randomizeUniform();
  MatrixPtr cpuT = cpu->getTranspose();
  MatrixPtr outputCheck = std::make_shared<CpuMatrix>(height, height);
  outputCheck->mul(*cpu, *cpuT);
  cpu->setDiag(1.0);
  cpu->add(*outputCheck);

  gpu->copyFrom(*cpu);
  cpu->inverse(cpuI, true);
  gpu->inverse(gpuI, false);

  TensorCheckErr(*cpuI, *gpuI);

  outputCheck->mul(*cpu, *cpuI);
  cpu->setDiag(1.0);
  TensorCheckErr(*cpu, *outputCheck);
}

TEST(Matrix, unary) {
  for (auto height : {1, 3, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 3, 32, 100, 512, 1000, 3210}) {
      VLOG(3) << " height=" << height << " width=" << width;

      testMatrixDeepSwap(height, width);
      testMatrixZeroAtOffset(height, width);
      testMatrixGetSum(height, width);
      testMatrixTranspose(height, width);
      testMatrixRotate(height, width);
    }
#ifdef LAPACK_FOUND
    // inverse matrix
    testMatrixInverse(height);
#else
    LOG(WARNING) << "Cannot run Matrix Inverse Unit Test.\n"
                 << "Failed to find lapack library in current system.\n"
                 << "To address this issue, Please adopt one of the following "
                    "approaches: \n"
                 << "1. Simply issue `sudo apt-get install liblapacke-dev` to "
                    "avoid re-build source code. \n"
                 << "2. Install MKL/Openblas/ATLAS and re-build PaddlePaddle "
                    "source code.";
#endif
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

  TensorCheckErr(*cpuOutput, *gpuOutput);
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

  TensorCheckErr(*cpuInput, *gpuInput);
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

  TensorCheckErr(*cpuOutput, *gpuOutput);
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

  TensorCheckErr(*cpuTable, *gpuTable);
}

TEST(Matrix, tableProjection) {
  for (auto numSamples : {10, 100, 1000, 10000, 80000}) {
    for (auto tableSize : {10, 100}) {
      for (auto inputDim : {20, 50}) {
        VLOG(3) << " numSamples=" << numSamples << " tableSize=" << tableSize
                << " inputDim=" << inputDim;
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

  cpuC->mul(*cpuA, *cpuB, alpha, beta);
  gpuC->mul(*gpuA, *gpuB, alpha, beta);

  TensorCheckErr(*cpuC, *gpuC);
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

  auto subMatrix = [](MatrixPtr& sub,
                      MatrixPtr matrix,
                      size_t startRow,
                      size_t endRow,
                      size_t startCol,
                      size_t endCol) {
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

  subCpuC->mul(*subCpuA, *subCpuB, alpha, beta);
  subGpuC->mul(*subGpuA, *subGpuB, alpha, beta);

  TensorCheckErr(*cpuC, *gpuC);
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

template <class T>
void testVectorReset(int size) {
  std::shared_ptr<CpuVectorT<T>> cpu = std::make_shared<CpuVectorT<T>>(size);
  std::shared_ptr<GpuVectorT<T>> gpu = std::make_shared<GpuVectorT<T>>(size);

  T value = (T)((int)rand() % 100 + 1.0f / ((int)rand() % 100));
  cpu->reset(value);
  gpu->reset(value);

  TensorCheckEqual(*cpu, *gpu);
}

template <class T>
void testVecortSelectFrom(int size) {
  std::shared_ptr<CpuVectorT<T>> cpuDst = std::make_shared<CpuVectorT<T>>(size);
  std::shared_ptr<GpuVectorT<T>> gpuDst = std::make_shared<GpuVectorT<T>>(size);
  std::shared_ptr<CpuVectorT<T>> cpuSrc =
      std::make_shared<CpuVectorT<T>>(size * 2);
  std::shared_ptr<GpuVectorT<T>> gpuSrc =
      std::make_shared<GpuVectorT<T>>(size * 2);
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

  TensorCheckEqual(*cpuDst, *gpuDst);
}

template <class T>
void testVecotrZeroMem(int size) {
  std::shared_ptr<CpuVectorT<T>> cpu = std::make_shared<CpuVectorT<T>>(size);
  std::shared_ptr<GpuVectorT<T>> gpu = std::make_shared<GpuVectorT<T>>(size);

  cpu->zeroMem();
  gpu->zeroMem();

  TensorCheckEqual(*cpu, *gpu);
}

template <class T>
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

  TensorCheckEqual(*cpuA, *gpuA);
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

  TensorCheckEqual(*cpuVal, *gpuVal);
}

TEST(Matrix, topK) {
  for (auto samples : {1, 5, 31, 90, 150, 500}) {
    for (auto dim :
         {1, 5, 8, 10, 15, 64, 80, 120, 256, 300, 1280, 5120, 50000}) {
      for (auto beamSize : {1, 5, 10, 20, 40, (int)rand() % dim + 1}) {
        if (beamSize > dim) continue;
        VLOG(3) << " samples=" << samples << " beamSize=" << beamSize
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

  TensorCheckEqual(*cpuVal, *gpuVal);

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
          VLOG(3) << " samples=" << samples << " beamSize=" << beamSize
                  << " dim=" << dim << " ratio=" << ratio;
          testSMatrixTopK(samples, dim, beamSize, ratio);
        }
      }
    }
  }
}

void testMatrixSequenceAvg(int batchSize, int inputDim, int mode) {
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

  TensorCheckErr(*cpuOutput, *gpuOutput);

  MatrixPtr cpuInGrad = std::make_shared<CpuMatrix>(batchSize, inputDim);
  MatrixPtr gpuInGrad = std::make_shared<GpuMatrix>(batchSize, inputDim);
  cpuInGrad->randomizeUniform();
  gpuInGrad->copyFrom(*cpuInGrad);

  cpuInGrad->sequenceAvgBackward(*cpuOutput, *cpuSequence, mode);
  gpuInGrad->sequenceAvgBackward(*gpuOutput, *gpuSequence, mode);

  TensorCheckErr(*cpuInGrad, *gpuInGrad);
}

TEST(Matrix, sequenceAvg) {
  for (auto batchSize : {10, 128, 6000}) {
    for (auto inputDim : {32, 100, 512}) {
      for (auto mode : {0, 1, 2}) {
        VLOG(3) << " batchSize=" << batchSize << " inputDim=" << inputDim
                << " mode=" << mode;
        testMatrixSequenceAvg(batchSize, inputDim, mode);
      }
    }
  }
}

void testParamReluBackwardDiff(int height,
                               int width,
                               int w_height,
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

  TensorCheckErr(*diff, *diffGpu);
}

TEST(Matrix, paramReluBackwardDiff) {
  for (auto height : {10, 40, 100}) {
    for (auto width : {10, 40, 100}) {
      for (auto w_height : {1, 2}) {
        for (auto w_width : {1, 2}) {
          if (width % (w_height * w_width)) continue;
          testParamReluBackwardDiff(height, width, w_height, w_width);
        }
      }
    }
  }
}

void testClassificationError(int numSamples, int dim, int topkSize) {
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

  cpuError->classificationError(*cpuOutput, *cpuLabel, topkSize);
  gpuError->classificationError(*gpuOutput, *gpuLabel, topkSize);

  TensorCheckEqual(*cpuError, *gpuError);
}

TEST(Matrix, classificationError) {
  for (auto numSamples : {1, 5, 31, 90, 150, 300}) {
    for (auto dim :
         {1, 5, 8, 10, 15, 64, 80, 120, 256, 300, 1280, 5120, 50000}) {
      for (auto topkSize : {1, 5, 10, 20, 40, (int)rand() % dim + 1}) {
        if (topkSize > dim) continue;
        VLOG(3) << " sample= " << numSamples << " topkSize= " << topkSize
                << " dim= " << dim;
        testClassificationError(numSamples, dim, topkSize);
      }
    }
  }
}

void testMaxPoolFwdBwd(int numSamples,
                       int channels,
                       int imgSizeH,
                       int imgSizeW,
                       int ksizeH,
                       int ksizeW,
                       int strideH,
                       int strideW,
                       int padH,
                       int padW) {
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

  target->maxPoolForward(*input,
                         imgSizeH,
                         imgSizeW,
                         channels,
                         ksizeW,
                         ksizeH,
                         strideH,
                         strideW,
                         outH,
                         outW,
                         padH,
                         padW);
  targetGpu->maxPoolForward(*inputGpu,
                            imgSizeH,
                            imgSizeW,
                            channels,
                            ksizeW,
                            ksizeH,
                            strideH,
                            strideW,
                            outH,
                            outW,
                            padH,
                            padW);
  MatrixPtr targetCheck = CpuMatrix::create(numSamples, outWidth, false, false);
  targetCheck->copyFrom(*targetGpu);
  checkMatrixEqual(target, targetCheck);

  MatrixPtr inputGrad = CpuMatrix::create(numSamples, inWidth, false, false);
  MatrixPtr inputGpuGrad = GpuMatrix::create(numSamples, inWidth, false, true);
  MatrixPtr targetGrad = CpuMatrix::create(numSamples, outWidth, false, false);
  MatrixPtr targetGpuGrad =
      GpuMatrix::create(numSamples, outWidth, false, true);

  inputGrad->randomizeUniform();
  targetGrad->randomizeUniform();
  inputGpuGrad->copyFrom(*inputGrad);
  targetGpuGrad->copyFrom(*targetGrad);

  inputGrad->maxPoolBackward(*input,
                             imgSizeH,
                             imgSizeW,
                             *targetGrad,
                             *target,
                             ksizeW,
                             ksizeH,
                             strideH,
                             strideW,
                             outH,
                             outW,
                             1.0,
                             1.0,
                             padH,
                             padW);
  inputGpuGrad->maxPoolBackward(*inputGpu,
                                imgSizeH,
                                imgSizeW,
                                *targetGpuGrad,
                                *targetGpu,
                                ksizeW,
                                ksizeH,
                                strideH,
                                strideW,
                                outH,
                                outW,
                                1.0,
                                1.0,
                                padH,
                                padW);
  MatrixPtr targetBwdCheck =
      CpuMatrix::create(numSamples, inWidth, false, false);
  targetBwdCheck->copyFrom(*inputGpuGrad);
  checkMatrixEqual(inputGrad, targetBwdCheck);
}

void testAvgPoolFwdBwd(int numSamples,
                       int channels,
                       int imgSizeH,
                       int imgSizeW,
                       int ksizeH,
                       int ksizeW,
                       int strideH,
                       int strideW,
                       int padH,
                       int padW) {
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

  target->avgPoolForward(*input,
                         imgSizeH,
                         imgSizeW,
                         channels,
                         ksizeW,
                         ksizeH,
                         strideH,
                         strideW,
                         outH,
                         outW,
                         padH,
                         padW);
  targetGpu->avgPoolForward(*inputGpu,
                            imgSizeH,
                            imgSizeW,
                            channels,
                            ksizeW,
                            ksizeH,
                            strideH,
                            strideW,
                            outH,
                            outW,
                            padH,
                            padW);

  TensorCheckErr(*target, *targetGpu);

  MatrixPtr inputGrad = CpuMatrix::create(numSamples, inWidth, false, false);
  MatrixPtr inputGpuGrad = GpuMatrix::create(numSamples, inWidth, false, true);
  MatrixPtr targetGrad = CpuMatrix::create(numSamples, outWidth, false, false);
  MatrixPtr targetGpuGrad =
      GpuMatrix::create(numSamples, outWidth, false, true);

  inputGrad->randomizeUniform();
  targetGrad->randomizeUniform();
  inputGpuGrad->copyFrom(*inputGrad);
  targetGpuGrad->copyFrom(*targetGrad);

  inputGrad->avgPoolBackward(*targetGrad,
                             imgSizeH,
                             imgSizeW,
                             ksizeW,
                             ksizeH,
                             strideH,
                             strideW,
                             outH,
                             outW,
                             1.0,
                             1.0,
                             padH,
                             padW);
  inputGpuGrad->avgPoolBackward(*targetGpuGrad,
                                imgSizeH,
                                imgSizeW,
                                ksizeW,
                                ksizeH,
                                strideH,
                                strideW,
                                outH,
                                outW,
                                1.0,
                                1.0,
                                padH,
                                padW);

  TensorCheckErr(*inputGrad, *inputGpuGrad);
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
                  for (auto pH : {0, (sizeY - 1) / 2}) {
                    for (auto pW : {0, (sizeX - 1) / 2}) {
                      VLOG(3) << " numSamples=" << numSamples
                              << " channels=" << channels
                              << " imgSizeH=" << imgSizeH
                              << " imgSizeW=" << imgSizeW << " sizeX=" << sizeX
                              << " sizeY=" << sizeY << " strideH=" << sH
                              << " strideW=" << sW << " padingH=" << pH
                              << " padingW=" << pW;
                      testMaxPoolFwdBwd(numSamples,
                                        channels,
                                        imgSizeH,
                                        imgSizeW,
                                        sizeX,
                                        sizeY,
                                        sH,
                                        sW,
                                        pH,
                                        pW);
                      testAvgPoolFwdBwd(numSamples,
                                        channels,
                                        imgSizeH,
                                        imgSizeW,
                                        sizeX,
                                        sizeY,
                                        sH,
                                        sW,
                                        pH,
                                        pW);
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

void testMaxOutFwdBwd(
    int numSamples, int imgSizeH, int imgSizeW, int channels, int groups) {
  int inWidth = imgSizeH * imgSizeW * channels;
  int outChannels = channels / groups;
  int outWidth = imgSizeH * imgSizeW * outChannels;

  // forward
  MatrixPtr input = CpuMatrix::create(numSamples, inWidth, false, false);
  MatrixPtr inputGpu = GpuMatrix::create(numSamples, inWidth, false, true);

  MatrixPtr target = CpuMatrix::create(numSamples, outWidth, false, false);
  MatrixPtr targetGpu = GpuMatrix::create(numSamples, outWidth, false, true);

  IVectorPtr id = CpuIVector::create(numSamples * outWidth, false);
  IVectorPtr idGpu = GpuIVector::create(numSamples * outWidth, true);

  input->randomizeUniform();
  inputGpu->copyFrom(*input);

  target->maxoutForward(*input, *id, outChannels, groups);
  targetGpu->maxoutForward(*inputGpu, *idGpu, outChannels, groups);

  TensorCheckErr(*target, *targetGpu);
  TensorCheckEqual(*id, *idGpu);

  // backward
  MatrixPtr inputGrad = CpuMatrix::create(numSamples, inWidth, false, false);
  MatrixPtr inputGpuGrad = GpuMatrix::create(numSamples, inWidth, false, true);

  MatrixPtr targetGrad = CpuMatrix::create(numSamples, outWidth, false, false);
  MatrixPtr targetGpuGrad =
      GpuMatrix::create(numSamples, outWidth, false, true);

  inputGrad->randomizeUniform();
  targetGrad->randomizeUniform();
  inputGpuGrad->copyFrom(*inputGrad);
  targetGpuGrad->copyFrom(*targetGrad);

  inputGrad->maxoutBackward(*targetGrad, *id, outChannels, groups);
  inputGpuGrad->maxoutBackward(*targetGpuGrad, *idGpu, outChannels, groups);

  TensorCheckErr(*inputGrad, *inputGpuGrad);
}

TEST(Matrix, MaxOutFwdBwd) {
  for (auto numSamples : {5, 10}) {
    for (auto channels : {8, 16}) {
      for (auto imgSizeH : {14, 28}) {
        for (auto imgSizeW : {16, 30}) {
          for (auto groups : {2, 4}) {
            VLOG(3) << " numSamples=" << numSamples << " channels=" << channels
                    << " imgSizeH=" << imgSizeH << " imgSizeW=" << imgSizeW
                    << " groups=" << groups;
            testMaxOutFwdBwd(numSamples, imgSizeH, imgSizeW, channels, groups);
          }
        }
      }
    }
  }
}

#endif
