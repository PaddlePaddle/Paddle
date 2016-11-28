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
/**
 * This test file compares the implementation of CPU and GPU function
 * in BaseMatrix.cpp.
 */

#include <gtest/gtest.h>
#include "paddle/utils/Util.h"
#include "paddle/math/BaseMatrix.h"
#include "TestUtils.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

/**
 * Test member functions which prototype is
 * void (BaseMatrix::*)().
 */
TEST(BaseMatrix, void) {
  typedef void (BaseMatrix::*FunctionProto)();
  #define BASEMATRIXCOMPARE(function) \
    BaseMatrixCompare(static_cast<FunctionProto>(&BaseMatrix::function));

  BASEMATRIXCOMPARE(neg);
  BASEMATRIXCOMPARE(exp);
  BASEMATRIXCOMPARE(log);
  BASEMATRIXCOMPARE(sqrt);
  BASEMATRIXCOMPARE(square);
  BASEMATRIXCOMPARE(reciprocal);
  BASEMATRIXCOMPARE(abs);
  BASEMATRIXCOMPARE(sign);
  BASEMATRIXCOMPARE(zero);
  BASEMATRIXCOMPARE(one);

  #undef BASEMATRIXCOMPARE
}

/**
 * Test member functions which prototype is
 * void (BaseMatrix::*)(real).
 */
TEST(BaseMatrix, real) {
  typedef void (BaseMatrix::*FunctionProto)(real);
  #define BASEMATRIXCOMPARE(function) \
    BaseMatrixCompare<0>(static_cast<FunctionProto>(&BaseMatrix::function));

  BASEMATRIXCOMPARE(pow);
  BASEMATRIXCOMPARE(subScalar);
  BASEMATRIXCOMPARE(mulScalar);
  BASEMATRIXCOMPARE(divScalar);
  BASEMATRIXCOMPARE(assign);
  BASEMATRIXCOMPARE(add);
  BASEMATRIXCOMPARE(biggerThanScalar);
  BASEMATRIXCOMPARE(downClip);

  #undef BASEMATRIXCOMPARE
}

/**
 * Test member functions which prototype is
 * void (BaseMatrix::*)(real, real).
 */
TEST(BaseMatrix, real_real) {
  typedef void (BaseMatrix::*FunctionProto)(real, real);
  #define BASEMATRIXCOMPARE(function) \
    BaseMatrixCompare<0, 1>(static_cast<FunctionProto>(&BaseMatrix::function));

  BASEMATRIXCOMPARE(add);
  BASEMATRIXCOMPARE(clip);

  #undef BASEMATRIXCOMPARE
}

/**
 * Test member functions which prototype is
 * void (BaseMatrix::*)(BaseMatrix&).
 */
TEST(BaseMatrix, BaseMatrix) {
  typedef void (BaseMatrix::*FunctionProto)(BaseMatrix&);
  #define BASEMATRIXCOMPARE(function) \
    BaseMatrixCompare<0>(static_cast<FunctionProto>(&BaseMatrix::function));

  BASEMATRIXCOMPARE(assign);
  BASEMATRIXCOMPARE(add);
  BASEMATRIXCOMPARE(relu);
  BASEMATRIXCOMPARE(reluDerivative);
  BASEMATRIXCOMPARE(softrelu);
  BASEMATRIXCOMPARE(softreluDerivative);
  BASEMATRIXCOMPARE(brelu);
  BASEMATRIXCOMPARE(breluDerivative);
  BASEMATRIXCOMPARE(square);
  BASEMATRIXCOMPARE(squareDerivative);
  BASEMATRIXCOMPARE(tanh);
  BASEMATRIXCOMPARE(tanhDerivative);

  BASEMATRIXCOMPARE(reciprocal);
  BASEMATRIXCOMPARE(reciprocalDerivative);
  BASEMATRIXCOMPARE(abs);
  BASEMATRIXCOMPARE(absDerivative);
  BASEMATRIXCOMPARE(sigmoid);
  BASEMATRIXCOMPARE(sigmoidDerivative);
  BASEMATRIXCOMPARE(expDerivative);
  BASEMATRIXCOMPARE(sign);
  BASEMATRIXCOMPARE(exp);
  BASEMATRIXCOMPARE(log);
  BASEMATRIXCOMPARE(sqrt);
  BASEMATRIXCOMPARE(dotMul);
  BASEMATRIXCOMPARE(dotMulSquare);
  BASEMATRIXCOMPARE(dotSquareMul);

  BASEMATRIXCOMPARE(addColVector);
  BASEMATRIXCOMPARE(addRowVector);
  BASEMATRIXCOMPARE(mulRowVector);
  BASEMATRIXCOMPARE(divRowVector);
  BASEMATRIXCOMPARE(addP2P);
  BASEMATRIXCOMPARE(invSqrt);

  #undef BASEMATRIXCOMPARE
}

/**
 * Test member functions which prototype is
 * void (BaseMatrix::*)(BaseMatrix&, real).
 */
TEST(BaseMatrix, BaseMatrix_real) {
  typedef void (BaseMatrix::*FunctionProto)(BaseMatrix&, real);
  #define BASEMATRIXCOMPARE(function) \
    BaseMatrixCompare<0, 1>(static_cast<FunctionProto>(&BaseMatrix::function));

  BASEMATRIXCOMPARE(addBias);
  BASEMATRIXCOMPARE(add);
  BASEMATRIXCOMPARE(sub);
  BASEMATRIXCOMPARE(pow);
  BASEMATRIXCOMPARE(addScalar);
  BASEMATRIXCOMPARE(subScalar);
  BASEMATRIXCOMPARE(mulScalar);
  BASEMATRIXCOMPARE(divScalar);
  BASEMATRIXCOMPARE(scalarDiv);
  BASEMATRIXCOMPARE(addSquare);

  BASEMATRIXCOMPARE(isEqualTo);

  #undef BASEMATRIXCOMPARE
}

/**
 * Test member functions which prototype is
 * void (BaseMatrix::*)(BaseMatrix&, BaseMatrix&).
 */
TEST(BaseMatrix, BaseMatrix_BaseMatrix) {
  typedef void (BaseMatrix::*FunctionProto)(BaseMatrix&, BaseMatrix&);
  #define BASEMATRIXCOMPARE(function) \
    BaseMatrixCompare<0, 1>(static_cast<FunctionProto>(&BaseMatrix::function));

  BASEMATRIXCOMPARE(softCrossEntropy);
  BASEMATRIXCOMPARE(softCrossEntropyBp);
  BASEMATRIXCOMPARE(binaryLabelCrossEntropy);
  BASEMATRIXCOMPARE(binaryLabelCrossEntropyBp);
  BASEMATRIXCOMPARE(sub);
  BASEMATRIXCOMPARE(add2);
  BASEMATRIXCOMPARE(dotMul);
  BASEMATRIXCOMPARE(dotDiv);
  BASEMATRIXCOMPARE(logisticRegressionLoss);
  BASEMATRIXCOMPARE(logisticRegressionLossBp);
  BASEMATRIXCOMPARE(biggerThan);
  BASEMATRIXCOMPARE(max);
  BASEMATRIXCOMPARE(dotMulSquare);
  BASEMATRIXCOMPARE(dotSquareSquare);

  #undef BASEMATRIXCOMPARE
}

/**
 * Test aggregate member functions which prototype is
 * void (BaseMatrix::*)(BaseMatrix&).
 */
TEST(Aggregate, BaseMatrix) {
  typedef void (BaseMatrix::*FunctionProto)(BaseMatrix&);
  #define BASEMATRIXAPPLYROW(function) \
    BaseMatrixApplyRow<0>(static_cast<FunctionProto>(&BaseMatrix::function));

 #define BASEMATRIXAPPLYCOL(function) \
    BaseMatrixApplyCol<0>(static_cast<FunctionProto>(&BaseMatrix::function));

  BASEMATRIXAPPLYROW(maxRows);
  BASEMATRIXAPPLYROW(minRows);

  BASEMATRIXAPPLYCOL(sumCols);
  BASEMATRIXAPPLYCOL(maxCols);
  BASEMATRIXAPPLYCOL(minCols);

  #undef BASEMATRIXAPPLYROW
  #undef BASEMATRIXAPPLYCOL
}

/**
 * Test aggregate member functions which prototype is
 * void (BaseMatrix::*)(BaseMatrix&, BaseMatrix&).
 */
TEST(Aggregate, BaseMatrix_BaseMatrix) {
  typedef void (BaseMatrix::*FunctionProto)(BaseMatrix&, BaseMatrix&);
  #define BASEMATRIXAPPLYROW(function) \
    BaseMatrixApplyRow<0, 1>(static_cast<FunctionProto>(&BaseMatrix::function));

  #define BASEMATRIXAPPLYCOL(function) \
    BaseMatrixApplyCol<0, 1>(static_cast<FunctionProto>(&BaseMatrix::function));

  BASEMATRIXAPPLYCOL(addDotMulVMM);

  #undef BASEMATRIXAPPLYROW
  #undef BASEMATRIXAPPLYCOL
}

/**
 * Test aggregate member functions which prototype is
 * void (BaseMatrix::*)(BaseMatrix&, real, real).
 */
TEST(Aggregate, BaseMatrix_real_real) {
  typedef void (BaseMatrix::*FunctionProto)(BaseMatrix&, real, real);
  #define BASEMATRIXAPPLYROW(function) \
    BaseMatrixApplyRow<0, 1, 2>(\
      static_cast<FunctionProto>(&BaseMatrix::function));

  #define BASEMATRIXAPPLYCOL(function) \
    BaseMatrixApplyCol<0, 1, 2>(\
      static_cast<FunctionProto>(&BaseMatrix::function));

  BASEMATRIXAPPLYROW(sumRows);
  BASEMATRIXAPPLYCOL(sumCols);

  #undef BASEMATRIXAPPLYROW
  #undef BASEMATRIXAPPLYCOL
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  return RUN_ALL_TESTS();
}

#endif
