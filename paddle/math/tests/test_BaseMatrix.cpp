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
 * This test file use autotest::AutoCompare and cmpWithoutArg to compares the
 * implementation of CPU and GPU member function in
 * BaseMatrix.cpp and Matrix.cpp.
 */

#include <gtest/gtest.h>
#include "TestUtils.h"
#include "paddle/math/BaseMatrix.h"

using paddle::BaseMatrix;
using paddle::Matrix;
using autotest::AutoCompare;

// Test all void (BaseMatrix::*)() function
TEST(BaseMatrix, void) {
  for (auto height : {1, 3, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 3, 32, 100, 512, 1000, 3210}) {
      auto compare = [height, width](void (BaseMatrix::*f)()) {
        AutoCompare test(height, width, 1e-5);
        test.cmpWithoutArg(f, height, width);
      };

      compare(&BaseMatrix::neg);
      compare(&BaseMatrix::exp2);
      compare(&BaseMatrix::log2);
      compare(&BaseMatrix::sqrt2);
      compare(&BaseMatrix::square2);
      compare(&BaseMatrix::reciprocal2);
      compare(&BaseMatrix::abs2);
      compare(&BaseMatrix::sign2);
      compare(&BaseMatrix::zero);
      compare(&BaseMatrix::one);
    }
  }
}

// Test all void (BaseMatrix::*)(real) function
TEST(BaseMatrix, real) {
  for (auto height : {1, 3, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 3, 32, 100, 512, 1000, 3210}) {
      auto compare = [height, width](void (BaseMatrix::*f)(real)) {
        AutoCompare test(height, width, 1e-5);
        test.cmpWithoutArg<0>(f, height, width);
      };

      compare(&BaseMatrix::pow2);
      compare(&BaseMatrix::subScalar);
      compare(&BaseMatrix::mulScalar);
      compare(&BaseMatrix::divScalar);
      compare(&BaseMatrix::assign);
      compare(&BaseMatrix::add);
      compare(&BaseMatrix::biggerThanScalar);
      compare(&BaseMatrix::downClip);
    }
  }
}

// Test all void (BaseMatrix::*)(BaseMatrix&) function
TEST(BaseMatrix, BaseMatrix) {
  for (auto height : {1, 3, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 3, 32, 100, 512, 1000, 3210}) {
      auto compare = [height, width](void (BaseMatrix::*f)(BaseMatrix&)) {
        AutoCompare test(height, width, 1e-5);
        test.cmpWithoutArg<0>(f, height, width);
      };

      compare(&BaseMatrix::assign);
      compare(&BaseMatrix::add);
      compare(&BaseMatrix::relu);
      compare(&BaseMatrix::reluDerivative);
      compare(&BaseMatrix::softrelu);
      compare(&BaseMatrix::softreluDerivative);
      compare(&BaseMatrix::brelu);
      compare(&BaseMatrix::breluDerivative);
      compare(&BaseMatrix::square2);
      compare(&BaseMatrix::squareDerivative);
      compare(&BaseMatrix::tanh);
      compare(&BaseMatrix::tanhDerivative);
      compare(&BaseMatrix::reciprocal2);
      compare(&BaseMatrix::reciprocalDerivative);
      compare(&BaseMatrix::abs2);
      compare(&BaseMatrix::absDerivative);
      compare(&BaseMatrix::sigmoid);
      compare(&BaseMatrix::sigmoidDerivative);
      compare(&BaseMatrix::expDerivative);
      compare(&BaseMatrix::sign2);
      compare(&BaseMatrix::exp2);
      compare(&BaseMatrix::log2);
      compare(&BaseMatrix::sqrt2);
      compare(&BaseMatrix::dotMul);
      compare(&BaseMatrix::dotMulSquare);
      compare(&BaseMatrix::dotSquareMul);
      compare(&BaseMatrix::addColVector);
      compare(&BaseMatrix::addRowVector);
      compare(&BaseMatrix::mulRowVector);
      compare(&BaseMatrix::divRowVector);
      compare(&BaseMatrix::mulColVector);
      compare(&BaseMatrix::divColVector);
      compare(&BaseMatrix::addP2P);
      compare(&BaseMatrix::invSqrt);
    }
  }
}

// Test all void (BaseMatrix::*)(real, real) function
TEST(BaseMatrix, real_real) {
  for (auto height : {1, 3, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 3, 32, 100, 512, 1000, 3210}) {
      auto compare = [height, width](void (BaseMatrix::*f)(real, real)) {
        AutoCompare test(height, width, 1e-5);
        test.cmpWithoutArg<0, 1>(f, height, width);
      };

      compare(&BaseMatrix::add);
      compare(&BaseMatrix::clip);
    }
  }
}

// Test all void (BaseMatrix::*)(BaseMatrix&, real) function
TEST(BaseMatrix, BaseMatrix_real) {
  for (auto height : {1, 3, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 3, 32, 100, 512, 1000, 3210}) {
      auto compare = [height, width](void (BaseMatrix::*f)(BaseMatrix&, real)) {
        AutoCompare test(height, width, 1e-5);
        test.cmpWithoutArg<0, 1>(f, height, width);
      };

      compare(&BaseMatrix::addBias);
      compare(&BaseMatrix::add);
      compare(&BaseMatrix::sub);
      compare(&BaseMatrix::pow2);
      compare(&BaseMatrix::addScalar);
      compare(&BaseMatrix::subScalar);
      compare(&BaseMatrix::mulScalar);
      compare(&BaseMatrix::divScalar);
      compare(&BaseMatrix::scalarDiv);
      compare(&BaseMatrix::addSquare);
      compare(&BaseMatrix::isEqualTo);
    }
  }
}

// Test all void (BaseMatrix::*)(BaseMatrix&, BaseMatrix&) function
TEST(BaseMatrix, BaseMatrix_BaseMatrix) {
  for (auto height : {1, 3, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 3, 32, 100, 512, 1000, 3210}) {
      auto compare = [height,
                      width](void (BaseMatrix::*f)(BaseMatrix&, BaseMatrix&)) {
        AutoCompare test(height, width, 1e-5);
        test.cmpWithoutArg<0, 1>(f, height, width);
      };

      compare(&BaseMatrix::softCrossEntropy);
      compare(&BaseMatrix::softCrossEntropyBp);
      compare(&BaseMatrix::binaryLabelCrossEntropy);
      compare(&BaseMatrix::binaryLabelCrossEntropyBp);
      compare(&BaseMatrix::sub);
      compare(&BaseMatrix::add2);
      compare(&BaseMatrix::dotMul);
      compare(&BaseMatrix::dotDiv);
      compare(&BaseMatrix::logisticRegressionLoss);
      compare(&BaseMatrix::logisticRegressionLossBp);
      compare(&BaseMatrix::biggerThan);
      compare(&BaseMatrix::max2);
      compare(&BaseMatrix::dotMulSquare);
      compare(&BaseMatrix::dotSquareSquare);
    }
  }
}

void TestEelementWise(size_t height, size_t width) {
  AutoCompare rowScale(height, width);
  rowScale.cmpWithoutArg<0, 1, 2>(&BaseMatrix::rowScale, height, width);

  AutoCompare rowDotMul(height, width);
  rowDotMul.cmpWithoutArg<0, 1, 2>(&BaseMatrix::rowDotMul, height, width);

  AutoCompare binaryClassificationError(height, width);
  binaryClassificationError.cmpWithoutArg<0, 1, 2, 3>(
      &BaseMatrix::binaryClassificationError, height, width);

  AutoCompare sumOfSquaresBp(height, width);
  sumOfSquaresBp.cmpWithoutArg<0, 1>(&Matrix::sumOfSquaresBp, height, width);
}

void TestAggregateToRow(size_t height, size_t width) {
  AutoCompare maxCols(1, width);
  maxCols.cmpWithoutArg<0>(&BaseMatrix::maxCols, height, width);

  AutoCompare minCols(1, width);
  minCols.cmpWithoutArg<0>(&BaseMatrix::minCols, height, width);

  AutoCompare addDotMulVMM(1, width);
  addDotMulVMM.cmpWithoutArg<0, 1>(&BaseMatrix::addDotMulVMM, height, width);

  AutoCompare sumCols(1, width);
  sumCols.cmpWithoutArg<0, 1, 2>(&BaseMatrix::sumCols, height, width);

  AutoCompare collectBias(1, width);
  collectBias.cmpWithoutArg<0, 1>(
      static_cast<void (Matrix::*)(Matrix&, real)>(&Matrix::collectBias),
      height,
      width);
}

void TestAggregateToCol(size_t height, size_t width) {
  AutoCompare maxRows(height, 1);
  maxRows.cmpWithoutArg<0>(&BaseMatrix::maxRows, height, width);

  AutoCompare minRows(height, 1);
  minRows.cmpWithoutArg<0>(&BaseMatrix::minRows, height, width);

  AutoCompare sumRows(height, 1);
  sumRows.cmpWithoutArg<0, 1, 2>(&BaseMatrix::sumRows, height, width);

  AutoCompare sumOfSquares(height, 1);
  sumOfSquares.cmpWithoutArg<0, 1>(&Matrix::sumOfSquares, height, width);
}

TEST(BaseMatrix, Other) {
  for (auto height : {1, 3, 11, 73, 128, 200, 330}) {
    for (auto width : {1, 3, 32, 100, 512, 1000, 3210}) {
      TestEelementWise(height, width);
      TestAggregateToRow(height, width);
      TestAggregateToCol(height, width);
    }
  }
}

#endif
