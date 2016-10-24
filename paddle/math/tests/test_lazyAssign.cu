/**
 * test_lazyAssign.cpp
 *
 * Author: hedaoyuan (hedaoyuan@baidu.com)
 * Created on: 2016-10-15
 *
 * Copyright (c) Baidu.com, Inc. All Rights Reserved
 */

#include <gtest/gtest.h>
#include "paddle/math/Matrix.h"
#include "paddle/math/TensorAssign.h"
#include "TensorCheck.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

typedef std::function<void(int height, int width)> testMatrixFunc;
void testMatrixCase(testMatrixFunc matrixFunc) {
  for (auto height : {1}) {
    for (auto width : {1, 32, 64, 128, 512, 1024, 4096, 32768, 65536, 131072,
                       262144, 524288, 1048576, 2097152, 4194304, 8388608}) {
      matrixFunc(height, width);
    }
  }
}

template<typename Tensor>
void testLazyAssign(int height, int width) {
  Tensor A1(height, width);
  Tensor A2(height, width);
  Tensor B(height, width);
  Tensor C(height, width);
  Tensor D(height, width);
  A1.randomizeUniform();
  B.randomizeUniform();
  C.randomizeUniform();
  D.randomizeUniform();
  A2.copyFrom(A1);

  EXPRESSION_PERFORMANCE(A1 = B + C; A1 = A1 * D;);

  EXPRESSION_PERFORMANCE(
    auto expr1 = A2.lazyAssign(B + C);
    auto expr2 = A2.lazyAssign(A2 * D);
    AssignEvaluate(expr1, expr2););

  TensorCheckErr(A1, A2);
}

TEST(lazyAssign, CPU) {
  testMatrixCase(testLazyAssign<CpuMatrix>);
}

#ifndef PADDLE_ONLY_CPU
TEST(lazyAssign, GPU) {
  testMatrixCase(testLazyAssign<GpuMatrix>);
}
#endif

template<typename Tensor>
void sgdUpdateTensor(Tensor& A, Tensor& B, Tensor& C, Tensor& D,
     real p1, real p2, real p3) {
  C = C * p2 - D * (B + A * p3) * p1;
  A += C;
}

void sgdUpdateLazyAssign(BaseMatrix& A, BaseMatrix& B,
    BaseMatrix& C, BaseMatrix& D,
    real p1, real p2, real p3) {
  auto expr1 = C.lazyAssign(C * p2 - D * (B + A * p3) * p1);
  auto expr2 = A.lazyAssign(A + C);
  AssignEvaluate(expr1, expr2);
}

template<typename Tensor>
void testSgdUpdate(int height, int width) {
  Tensor A1(height, width);
  Tensor A2(height, width);
  Tensor A3(height, width);
  A1.randomizeUniform();
  A2.copyFrom(A1);
  A3.copyFrom(A1);

  Tensor B(height, width);
  B.randomizeUniform();

  Tensor C1(height, width);
  Tensor C2(height, width);
  Tensor C3(height, width);
  C1.randomizeUniform();
  C2.copyFrom(C1);
  C3.copyFrom(C1);

  Tensor D(height, width);
  D.randomizeUniform();

  real p1 = 0.2;
  real p2 = 0.3;
  real p3 = 0.5;

  /**
   * c = p2 * c - p1 * (b + p3 * a);
   * a = a + c;
   */
  // BaseMatrix API
  EXPRESSION_PERFORMANCE(
  A1.sgdUpdate(B, C1, D, p1, p2, p3););

  // Tensor expression
  EXPRESSION_PERFORMANCE(
    sgdUpdateTensor(A2, B, C2, D, p1, p2, p3));

  // lazyAssign
  EXPRESSION_PERFORMANCE(
    sgdUpdateLazyAssign(A3, B, C3, D, p1, p2, p3));

  TensorCheckErr(A1, A2);
  TensorCheckErr(A1, A3);
  TensorCheckErr(C1, C2);
  TensorCheckErr(C1, C3);
}

TEST(sgdUpdate, CPU) {
  testMatrixCase(testSgdUpdate<CpuMatrix>);
}

#ifndef PADDLE_ONLY_CPU
TEST(sgdUpdate, GPU) {
  testMatrixCase(testSgdUpdate<GpuMatrix>);
}
#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  hl_start();
  hl_init(0);
  return RUN_ALL_TESTS();
}

