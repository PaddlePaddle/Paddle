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
#include "paddle/math/Matrix.h"
#include "TensorCheck.h"

using paddle::Matrix;
using paddle::CpuMatrix;
using paddle::GpuMatrix;
using paddle::CpuVector;
using paddle::GpuVector;
using paddle::CpuIVector;
using paddle::GpuIVector;
using autotest::TensorCheckEqual;
using autotest::TensorCheckErr;

#define INIT_UNARY(A1, A2)                  \
    Tensor A1(height, width);               \
    Tensor A2(height, width);               \
    A1.randomizeUniform();                  \
    A2.copyFrom(A1)
#define INIT_BINARY(A1, A2, B)              \
    INIT_UNARY(A1, A2);                     \
    Tensor B(height, width);                \
    B.randomizeUniform()
#define INIT_TERNARY(A1, A2, B, C)          \
    INIT_BINARY(A1, A2, B);                 \
    Tensor C(height, width);                \
    C.randomizeUniform()
#define INIT_QUATERNARY(A1, A2, B, C, D)    \
    INIT_TERNARY(A1, A2, B, C);             \
    Tensor D(height, width);                \
    D.randomizeUniform()

template<typename Tensor>
struct TestUnaryMatrix {
  typedef std::function<void(Tensor& A1, Tensor& A2)> UnaryFunc;

  explicit TestUnaryMatrix(UnaryFunc testUnaryFunc) {
    for (auto height : {1, 11, 73, 128, 200, 330}) {
      for (auto width : {1, 32, 100, 512, 1000, 3210}) {
        LOG(INFO) << " height=" << height << " width=" << width;
        INIT_UNARY(A1, A2);
        testUnaryFunc(A1, A2);
      }
    }
  }
};

template<typename Tensor>
struct TestBinaryMatrix {
  typedef std::function<void(Tensor& A1, Tensor& A2, Tensor& B)> BinaryFunc;

  explicit TestBinaryMatrix(BinaryFunc testBinaryFunc) {
    for (auto height : {1, 11, 73, 128, 200, 330}) {
      for (auto width : {1, 32, 100, 512, 1000, 3210}) {
        LOG(INFO) << " height=" << height << " width=" << width;
        INIT_BINARY(A1, A2, B);
        testBinaryFunc(A1, A2, B);
      }
    }
  }
};

template<typename Tensor>
struct TestTernaryMatrix {
  typedef std::function<void(
    Tensor& A1, Tensor& A2, Tensor& B, Tensor& C)> TernaryFunc;

  explicit TestTernaryMatrix(TernaryFunc testTernaryFunc) {
    for (auto height : {1, 11, 73, 128, 200, 330}) {
      for (auto width : {1, 32, 100, 512, 1000, 3210}) {
        LOG(INFO) << " height=" << height << " width=" << width;
        INIT_TERNARY(A1, A2, B, C);
        testTernaryFunc(A1, A2, B, C);
      }
    }
  }
};

template<typename Tensor>
struct TestQuaternaryMatrix {
  typedef std::function<void(
    Tensor& A1, Tensor& A2, Tensor& B, Tensor& C, Tensor& D)> QuaternaryFunc;

  explicit TestQuaternaryMatrix(QuaternaryFunc testQuaternaryFunc) {
    for (auto height : {1, 11, 73, 128, 200, 330}) {
      for (auto width : {1, 32, 100, 512, 1000, 3210}) {
        LOG(INFO) << " height=" << height << " width=" << width;
        INIT_QUATERNARY(A1, A2, B, C, D);
        testQuaternaryFunc(A1, A2, B, C, D);
      }
    }
  }
};

template<typename Tensor, class T>
struct TestUnaryVectorT {
  typedef std::function<void(Tensor& A1, Tensor& A2)> UnaryFunc;

  explicit TestUnaryVectorT(UnaryFunc testUnaryFunc) {
    for (auto size : {1, 11, 73, 128, 200, 330, 512, 1000, 4210}) {
      LOG(INFO) << " size=" << size;
      Tensor A1(size);
      Tensor A2(size);
      if (typeid(T) == typeid(real)) {
        A1.rand();
      } else {
        A1.rand(1000);
      }
      A2.copyFrom(A1);
      testUnaryFunc(A1, A2);
    }
  }
};

void SetTensorValue(Matrix& matrix, real value) {
  int height = matrix.getHeight();
  int width = matrix.getWidth();
  int stride = matrix.getStride();
  real* data = matrix.getData();
  for (int i = 0; i < height; i++) {
    int j = rand() % width;  // NOLINT
    if (typeid(matrix) == typeid(CpuMatrix)) {
      data[i * stride + j] = value;
    } else if (typeid(matrix) == typeid(GpuMatrix)) {
      hl_memcpy(&data[i * stride + j], &value, sizeof(real));
    } else {
    }
  }
}

template<typename Tensor>
void testTensorAddScalar(Tensor& A1, Tensor& A2) {
  real p1 = 2.5;
  real p2 = 3.0;
  A1.add(p1);   // a += p
  A2 += p1;
  TensorCheckEqual(A1, A2);

  A1.add(p1, p2);  // a = a * p1 + p2
  A2 = A2 * p1 + p2;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorSubScalar(Tensor& A1, Tensor& A2) {
  real p = 2.5;
  A1.subScalar(p);  // a -= p
  A2 -= p;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorMulScalar(Tensor& A1, Tensor& A2) {
  real p = 2.5;
  A1.mulScalar(p);  // a *= p
  A2 *= p;
  TensorCheckEqual(A1, A2);

  real learningRate = 0.7f;
  real decayRate = 1.2f;
  A1.applyL2(learningRate, decayRate);
  A2 = A2 * (1.0f / (1.0f + learningRate * decayRate));
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorDivScalar(Tensor& A1, Tensor& A2) {
  real p = 2.5;
  A1.divScalar(p);  // a /= p
  A2 /= p;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorNeg(Tensor& A1, Tensor& A2) {
  A1.neg();  // a = -a
  A2 = -A2;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorAbs(Tensor& A1, Tensor& A2) {
  A1.abs2();  // a = a > 0 ? a : -a
  A2 = A2.abs();
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorSquare(Tensor& A1, Tensor& A2) {
  A1.square2();  // a = a * a
  A2 = A2.square();
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorReciprocal(Tensor& A1, Tensor& A2) {
  A1.reciprocal2();  // a = 1.0f / a
  A2 = A2.reciprocal();
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorSign(Tensor& A1, Tensor& A2) {
  A1.sign2();  // a = (a > 0) - (a < 0)
  A2 = A2.sign();
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorAssign(Tensor& A1, Tensor& A2) {
  A1.assign(1.5);   // a = p
  A2 = A2.constant(1.5);
  TensorCheckEqual(A1, A2);

  A1.one();  // a = 1
  A2 = A2.constant(1.0);
  TensorCheckEqual(A1, A2);

  A1.zero();  // a = 0
  A2 = A2.constant(0.0);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testUnaryBaseOp(Tensor& A1, Tensor& A2) {
  testTensorAddScalar(A1, A2);
  testTensorSubScalar(A1, A2);
  testTensorMulScalar(A1, A2);
  testTensorDivScalar(A1, A2);
  testTensorNeg(A1, A2);
  testTensorAbs(A1, A2);
  testTensorSquare(A1, A2);
  testTensorReciprocal(A1, A2);
  testTensorSign(A1, A2);
  testTensorAssign(A1, A2);
}

template<typename Tensor>
void testUnaryBaseOpInt(Tensor& A1, Tensor& A2) {
  A1.add(2);   // a += p
  A2 += 2;
  TensorCheckEqual(A1, A2);

  A1.add(3, 2);  // a = a * p1 + p2
  A2 = A2 * 3 + 2;
  TensorCheckEqual(A1, A2);

  testTensorNeg(A1, A2);
  testTensorAbs(A1, A2);
}

TEST(Unary, BaseOp) {
  TestUnaryMatrix<CpuMatrix> testCpuMatrix(testUnaryBaseOp<CpuMatrix>);
  TestUnaryVectorT<CpuVector, real> testCpuVector(testUnaryBaseOp<CpuVector>);
  TestUnaryVectorT<CpuIVector, int>
    testCpuIVector(testUnaryBaseOpInt<CpuIVector>);

#ifndef PADDLE_ONLY_CPU
  TestUnaryMatrix<GpuMatrix> testGpuMatrix(testUnaryBaseOp<GpuMatrix>);
  TestUnaryVectorT<GpuVector, real> testGpuVector(testUnaryBaseOp<GpuVector>);
  TestUnaryVectorT<GpuIVector, int>
    testGpuIVector(testUnaryBaseOpInt<GpuIVector>);
#endif
}

template<typename Tensor>
void testTensorExp(Tensor& A1, Tensor& A2) {
  A1.exp2();  // a = exp(a)
  A2 = A2.exp();
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorLog(Tensor& A1, Tensor& A2) {
  A1.log2();  // a = log(a)
  A2 = A2.log();
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorSqrt(Tensor& A1, Tensor& A2) {
  A1.sqrt2();  // a = sqrt(a)
  A2 = A2.sqrt();
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorPow(Tensor& A1, Tensor& A2) {
  A1.pow2(3.2);  // a = pow(a, p)
  A2 = A2.pow(3.2);
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testUnayrMathOp(Tensor& A1, Tensor& A2) {
  testTensorExp(A1, A2);
  testTensorLog(A1, A2);
  testTensorSqrt(A1, A2);
  testTensorPow(A1, A2);
}

TEST(Unary, MathOp) {
  TestUnaryMatrix<CpuMatrix> testCpu(testUnayrMathOp<CpuMatrix>);

#ifndef PADDLE_ONLY_CPU
  TestUnaryMatrix<GpuMatrix> testGpu(testUnayrMathOp<GpuMatrix>);
#endif
}

template<typename Tensor>
void testTensorClip(Tensor& A1, Tensor& A2) {
  real p1 = 0.003f;
  real p2 = 0.877f;
  A1.clip(p1, p2);  // a = a < p1 ? p1 : (a > p2 ? p2 : a)
  // A2 = A2.min(0.877f).max(0.003f);
  A2 = (A2 < p1).condition(p1, (A2 > p2).condition(p2, A2));
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorBiggerThanScalar(Tensor& A1, Tensor& A2) {
  real p = 0.5f;
  A1.biggerThanScalar(p);  // a = a > p ? 1.0f : 0.0f
  A2 = (A2 > p).condition((real)1.0, (real)0.0);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorapplyL1(Tensor& A1, Tensor& A2) {
  /**
   * T lambda = p;
   * a = (a > lambda) ? (a - lambda)
   *                  : (a < -lambda) ? (a + lambda) : 0
   *
   * p = learningRate * decayRate;
   */
  real learningRate = 0.7f;
  real decayRate = 0.6f;
  A1.applyL1(learningRate, decayRate);
  A2 = (A2 > (learningRate * decayRate)).condition(
    (A2 - (learningRate * decayRate)),
    (A2 < -(learningRate * decayRate)).condition(
      (A2 + (learningRate * decayRate)), (real)0.0));
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testUnayrCompareOp(Tensor& A1, Tensor& A2) {
  testTensorClip(A1, A2);
  testTensorBiggerThanScalar(A1, A2);

  A1.randomizeUniform();
  A1.subScalar(0.5f);
  A2.copyFrom(A1);
  testTensorapplyL1(A1, A2);
}

TEST(Unary, CompareOp) {
  TestUnaryMatrix<CpuMatrix> testCpu(testUnayrCompareOp<CpuMatrix>);

#ifndef PADDLE_ONLY_CPU
  TestUnaryMatrix<GpuMatrix> testGpu(testUnayrCompareOp<GpuMatrix>);
#endif
}

template<typename Tensor>
void testTensorAdd(Tensor& A1, Tensor& A2, Tensor& B) {
  real p1 = 2.5;
  real p2 = 3.2;
  A1.add(B);  // a += b
  A2 += B;
  TensorCheckEqual(A1, A2);

  A1.add(B, p1);  // a += b * p
  A2 += B * p1;
  TensorCheckEqual(A1, A2);

  A1.add(B, p1, p2);  // a = p1 * a + p2 * b
  A2 = A2 * p1 + B * p2;
  TensorCheckEqual(A1, A2);

  A1.addScalar(B, p1);  // a = b + p
  A2 = B + p1;
  TensorCheckEqual(A1, A2);

  A1.addSquare(B, p1);  // a += p * b * b
  A2 += B.constant(p1) * B * B;
  TensorCheckEqual(A1, A2);

  A1.decayAddSquare(B, p1, p2);  // a = p1 * a + p2 * b * b
  A2 = A2 * p1 + B.constant(p2) * B * B;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorSub(Tensor& A1, Tensor& A2, Tensor& B) {
  real p = 2.5;
  A1.sub(B);  // a -= b
  A2 -= B;
  TensorCheckEqual(A1, A2);

  A1.sub(B, p);  // a -= b * p
  A2 -= B * p;
  TensorCheckEqual(A1, A2);

  A1.subScalar(B, p);  // a = b - p
  A2 = B - p;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorMul(Tensor& A1, Tensor& A2, Tensor& B) {
  real p = 2.5;
  A1.mulScalar(B, p);  // a = b * p
  A2 = B * p;
  TensorCheckEqual(A1, A2);

  A1.dotMulSquare(B);  // a *= b * b
  A2 *= B * B;
  TensorCheckEqual(A1, A2);

  A1.dotSquareMul(B);  // a = a * a * b
  A2 = A2 * A2 * B;
  TensorCheckEqual(A1, A2);

  A1.dotMul(B);  // a *= b
  A2 *= B;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorDiv(Tensor& A1, Tensor& A2, Tensor& B) {
  real p = 2.5;
  A1.divScalar(B, p);  // a = b / p
  A2 = B / p;
  TensorCheckEqual(A1, A2);

  A1.scalarDiv(B, p);  // a = p / b
  A2 = B.constant(p) / B;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorAssign(Tensor& A1, Tensor& A2, Tensor& B) {
  A1.assign(B);  // a = b
  A2 = B;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorSquare(Tensor& A1, Tensor& A2, Tensor& B) {
  B.square2(A1);   // b = a * a
  A2 = B.square();
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorSquareDerivative(Tensor& A1, Tensor& A2, Tensor& B) {
  A1.squareDerivative(B);  // a *= 2.0 * b
  A2 = A2 * (real)2.0 * B;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorReciprocal(Tensor& A1, Tensor& A2, Tensor& B) {
  B.reciprocal2(A1);  // b = 1.0f / a
  A2 = B.reciprocal();
  TensorCheckEqual(A1, A2);

  real p1 = 0.58;
  real p2 = 0.32;
  A1.reciprocal2(B, p1, p2);  // a = 1 / (p1 * b + p2)
  A2 = (B * p1 + p2).reciprocal();
  TensorCheckEqual(A1, A2);

  real learningRate = 0.7f;
  real decayRate = 1.2f;
  A1.applyL2(B, learningRate, decayRate);  // a *= (1.0f / (1.0f + p * b))
  A2 *= (B.constant(1.0f) +
    B.constant(learningRate * decayRate) * B).reciprocal();
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorReciprocalDerivative(Tensor& A1, Tensor& A2, Tensor& B) {
  A1.reciprocalDerivative(B);  // a *= -b * b
  A2 *= (-B) * B;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorSign(Tensor& A1, Tensor& A2, Tensor& B) {
  B.sign2(A1);  // b = a > 0.0f ? 1.0f : -1.0f
  A2 = B.sign();
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorAbs(Tensor& A1, Tensor& A2, Tensor& B) {
  B.abs2(A1);  // b = a > 0.0f ? a : -a
  A2 = B.abs();
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testBinaryBaseOp(Tensor& A1, Tensor& A2, Tensor& B) {
  testTensorAdd(A1, A2, B);
  testTensorSub(A1, A2, B);
  testTensorMul(A1, A2, B);
  testTensorDiv(A1, A2, B);
  testTensorSquare(A1, A2, B);
  testTensorSquareDerivative(A1, A2, B);
  testTensorReciprocal(A1, A2, B);
  testTensorReciprocalDerivative(A1, A2, B);
  testTensorAbs(A1, A2, B);
  testTensorSign(A1, A2, B);
  testTensorAssign(A1, A2, B);
}

TEST(Binary, BaseOp) {
  TestBinaryMatrix<CpuMatrix> testCpu(testBinaryBaseOp<CpuMatrix>);

#ifndef PADDLE_ONLY_CPU
  TestBinaryMatrix<GpuMatrix> testGpu(testBinaryBaseOp<GpuMatrix>);
#endif
}

template<typename Tensor>
void testTensorExp(Tensor& A1, Tensor& A2, Tensor& B) {
  // a = exp(b)
  A1.exp2(B);
  A2 = B.exp();
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorExpDerivative(Tensor& A1, Tensor& A2, Tensor& B) {
  A1.expDerivative(B);  // a *= b
  A2 *= B;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorLog(Tensor& A1, Tensor& A2, Tensor& B) {
  // a = log(b)
  A1.log2(B);
  A2 = B.log();
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorSqrt(Tensor& A1, Tensor& A2, Tensor& B) {
  // a = sqrt(b)
  A1.sqrt2(B);
  A2 = B.sqrt();
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorInvSqrt(Tensor& A1, Tensor& A2, Tensor& B) {
  // a = 1.0f / sqrt(b)
  A1.invSqrt(B);
  A2 = B.sqrt().reciprocal();
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorPow(Tensor& A1, Tensor& A2, Tensor& B) {
  A1.pow2(B, 2.5f);  // a = pow(b, p)
  A2 = B.pow(2.5f);
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorSoftrelu(Tensor& A1, Tensor& A2, Tensor& B) {
  /*
   * const T THRESHOLD = 40.0;
   * b = log(1.0 +
   *         exp((a > THRESHOLD) ? THRESHOLD
   *             : ((a < -THRESHOLD) ? (-THRESHOLD) : a)))
   */
  B.softrelu(A1);

  real THRESHOLD = 40.0;
  A2 = (B.constant(1.0f) +
        (B > THRESHOLD).condition(
          THRESHOLD, (B < -THRESHOLD).condition(-THRESHOLD, B)).exp()).log();
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorSoftreluDerivative(Tensor& A1, Tensor& A2, Tensor& B) {
  /*
   * const T THRESHOLD = 40.0;
   * a *= (1.0 - exp(-1.0 * ((b > THRESHOLD)
   *                             ? THRESHOLD
   *                             : ((b < -THRESHOLD) ? (-THRESHOLD) : b)))));
   */
  A1.softreluDerivative(B);
  real THRESHOLD = 40.0;
  A2 = A2 * (B.constant(1.0f) -
             (B.constant(-1.0f) *
              (B > THRESHOLD).condition(
                THRESHOLD, (B < -THRESHOLD).condition(-THRESHOLD, B))).exp());
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorSigmoid(Tensor& A1, Tensor& A2, Tensor& B) {
  /*
    const T THRESHOLD_MIN = -40.0;
    const T THRESHOLD_MAX = 13.0;
    T tmp = (a < THRESHOLD_MIN) ? THRESHOLD_MIN
            : ((a > THRESHOLD_MAX) ? THRESHOLD_MAX : a);
    b = 1.0f / (1.0f + exp(-tmp)))
   */
  B.sigmoid(A1);

  const real THRESHOLD_MIN = -40.0;
  const real THRESHOLD_MAX = 13.0;
  auto tmp = (B < THRESHOLD_MIN).condition(
    THRESHOLD_MIN, (B > THRESHOLD_MAX).condition(THRESHOLD_MAX, B));
  A2 = (B.constant(1.0f) + (-tmp).exp()).reciprocal();
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorSigmoidDerivative(Tensor& A1, Tensor& A2, Tensor& B) {
  A1.sigmoidDerivative(B);  // a *= b * (1 - b)
  A2 *= B * (B.constant(1.0f) - B);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorTanh(Tensor& A1, Tensor& A2, Tensor& B) {
  B.tanh(A1);  // b = 2.0 / (1.0 + exp(-2 * a)) - 1.0
  A2 = B.constant(2.0f) / ((B * ((real)-2.0f)).exp() + (real)1.0f) - (real)1.0f;
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorTanhDerivative(Tensor& A1, Tensor& A2, Tensor& B) {
  A1.tanhDerivative(B);  // a *= 1 - b * b
  A2 *= B.constant(1.0f) - B * B;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorScaledTanh(Tensor& A1, Tensor& A2, Tensor& B) {
  real p1 = 2.5;
  real p2 = 3.1;
  // b = p1 * (2.0 / (1.0 + exp(-2 * p2 * a)) - 1.0)
  B.scaledTanh(A1, p1, p2);
  A2 = B.constant(p1) *
      (B.constant(2.0f) / ((B.constant(-2.0f) * p2 * B).exp() + (real)1.0)
       - (real)1.0);
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorScaledTanhDerivative(Tensor& A1, Tensor& A2, Tensor& B) {
  real p1 = 2.5;
  real p2 = 3.1;
  // a *= (p2 / p1) * (p1 * p1 - b * b));
  A1.scaledTanhDerivative(B, p1, p2);
  A2 = A2 * (B.constant(p2 / p1) * (B.constant(p1 * p1) - B * B));
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testBinaryMathOp(Tensor& A1, Tensor& A2, Tensor& B) {
  testTensorTanhDerivative(A1, A2, B);
  testTensorScaledTanhDerivative(A1, A2, B);
  testTensorSigmoidDerivative(A1, A2, B);
  testTensorExpDerivative(A1, A2, B);
  testTensorScaledTanh(A1, A2, B);
  testTensorTanh(A1, A2, B);
  testTensorExp(A1, A2, B);
  testTensorLog(A1, A2, B);
  testTensorSqrt(A1, A2, B);
  testTensorInvSqrt(A1, A2, B);
  testTensorPow(A1, A2, B);

  testTensorSoftrelu(A1, A2, B);
  testTensorSoftreluDerivative(A1, A2, B);
  testTensorSigmoid(A1, A2, B);
}

TEST(Binary, MathOp) {
  TestBinaryMatrix<CpuMatrix> testCpu(testBinaryMathOp<CpuMatrix>);

#ifndef PADDLE_ONLY_CPU
  TestBinaryMatrix<GpuMatrix> testGpu(testBinaryMathOp<GpuMatrix>);
#endif
}

template<typename Tensor>
void testTensorRelu(Tensor& A1, Tensor& A2, Tensor& B) {
  B.relu(A1);  // b = a > 0.0f ? a : 0.0f
  A2 = (B > (real)0.0f).condition(B, (real)0.0f);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorReluDerivative(Tensor& A1, Tensor& A2, Tensor& B) {
  A1.reluDerivative(B);  // a *= (b > 0.0f ? 1.0f : 0.0f)
  A2 *= (B > (real)0.0).condition((real)1.0, (real)0.0);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorBrelu(Tensor& A1, Tensor& A2, Tensor& B) {
  /*
   * b = a > p1 ? a : p1
   * b = b < p2 ? b : p2
   * int p1 = 0, p2 = 24;
   */
  SetTensorValue(B, 32.0f);
  B.brelu(A1);
  auto tmp = (B > (real)0.0f).condition(B, (real)0.0f);
  A2 = (tmp < (real)24.0f).condition(tmp, (real)24.0f);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorBreluDerivative(Tensor& A1, Tensor& A2, Tensor& B) {
  SetTensorValue(B, 32.0f);
  /*
   * a *= (b > p1 && b < p2) ? 1.0 : 0.0
   * int p1 = 0, p2 = 24;
   */
  A1.breluDerivative(B);
  A2 *= (B > (real)0.0f && B < (real)24.0f).condition((real)1.0f, (real)0.0f);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorAbsDerivative(Tensor& A1, Tensor& A2, Tensor& B) {
  A1.absDerivative(B);  // a = (b > 0) ? a : (b < 0) ? -a : 0
  A2 = (B > (real)0.0f).condition(A2,
    (B < (real)0.0f).condition(-A2, (real)0.0f));
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorIsEqualTo(Tensor& A1, Tensor& A2, Tensor& B) {
  real p = 0.613;
  SetTensorValue(B, p);
  A1.isEqualTo(B, p);  // a = (b == p)
  A2 = (B == p);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorapplyL1(Tensor& A1, Tensor& A2, Tensor& B) {
  /**
   * T lambda = p * b;
   * a = (a > lambda) ? (a - lambda)
   *                  : (a < -lambda) ? (a + lambda) : 0
   *
   * p = learningRate * decayRate;
   */
  real learningRate = 0.7f;
  real decayRate = 0.6f;
  A1.applyL1(B, learningRate, decayRate);
  auto lambda = B.constant(learningRate * decayRate) * B;
  A2 = (A2 > lambda).condition(
    (A2 - lambda), (A2 < -lambda).condition((A2 + lambda), (real)0.0f));
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testBinaryCompareOp(Tensor& A1, Tensor& A2, Tensor& B) {
  B.subScalar(0.5f);
  SetTensorValue(B, 0.0f);
  testTensorReluDerivative(A1, A2, B);

  A1.randomizeUniform();
  A2.copyFrom(A1);
  testTensorBreluDerivative(A1, A2, B);

  testTensorAbsDerivative(A1, A2, B);
  testTensorRelu(A1, A2, B);
  testTensorBrelu(A1, A2, B);
  testTensorIsEqualTo(A1, A2, B);
}

TEST(Binary, CompareOp) {
  TestBinaryMatrix<CpuMatrix> testCpu(testBinaryCompareOp<CpuMatrix>);

#ifndef PADDLE_ONLY_CPU
  TestBinaryMatrix<GpuMatrix> testGpu(testBinaryCompareOp<GpuMatrix>);
#endif
}

template<typename Tensor>
void testTensorAdd(Tensor& A1, Tensor& A2, Tensor& B, Tensor& C) {
  A1.add(B, C);  // a = b + c
  A2 = B + C;
  TensorCheckEqual(A1, A2);

  real p1 = 1.5;
  real p2 = 2.5;
  real p3 = 3.8;
  A1.add(B, p1, C, p2);  // a = p1 * b + p2 * c
  A2 = B * p1 + C * p2;
  TensorCheckEqual(A1, A2);

  A1.add2(B, C);  // a = a + b + c
  A2 = A2 + B + C;
  TensorCheckEqual(A1, A2);

  A1.add2(B, C, p1, p2, p3);  // a = p1 * a + p2 * b + p3 * c
  A2 = A2 * p1 + B * p2 + C * p3;
  TensorCheckEqual(A1, A2);

  A1.decayAddSquareMul(B, C, p1, p2);  // a = p1 * a + p2 * b * b * c * c
  A2 = A2 * p1 + B.constant(p2) * B * B * C * C;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorSub(Tensor& A1, Tensor& A2, Tensor& B, Tensor& C) {
  A1.sub(B, C);  // a = b - c
  A2 = B - C;
  TensorCheckEqual(A1, A2);

  real p1 = 1.5;
  real p2 = 2.5;
  A1.sub(B, p1, C, p2);  // a = p1 * b - p2 * c
  A2 = B * p1 - C * p2;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorMul(Tensor& A1, Tensor& A2, Tensor& B, Tensor& C) {
  A1.dotMul(B, C);  // a = b * c
  A2 = B * C;
  TensorCheckEqual(A1, A2);

  A1.dotMulSquare(B, C);  // a = b * c * c
  A2 = B * C * C;
  TensorCheckEqual(A1, A2);

  A1.dotSquareSquare(B, C);  // a = b * b * c * c
  A2 = B * B * C * C;
  TensorCheckEqual(A1, A2);

  real p1 = 1.5;
  real p2 = 2.5;

  /*
   * T tmp = p1 * b + p2 * c;
   * a *= tmp * tmp
   */
  A1.dotMulSquareSum(B, C, p1, p2);
  auto tmp = B * p1 + C * p2;
  A2 *= tmp * tmp;
  TensorCheckEqual(A1, A2);

  /*
   * T tmp = p1 * b + p2 * c;
   * a = tmp * tmp
   */
  A1.dotSquareSum(B, C, p1, p2);
  auto tmp2 = B * p1 + C * p2;
  A2 = tmp2 * tmp2;
  TensorCheckEqual(A1, A2);

  // a *= p1 * b + p2 * c
  A1.dotMulSum(B, C, p1, p2);
  A2 *= B * p1 + C * p2;
  TensorCheckEqual(A1, A2);

  // a = p1 * a + p2 * b * c
  A1.addDotMul(B, C, p1, p2);
  A2 = A2 * p1 + B.constant(p2) * B * C;
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorDiv(Tensor& A1, Tensor& A2, Tensor& B, Tensor& C) {
  A1.dotDiv(B, C);  // a = (b == 0.0) ? 0.0 : b / c
  A2 = (B == (real)0.0).condition((real)0.0, B / C);
  TensorCheckEqual(A1, A2);

  real p1 = 1.5;
  real p2 = 2.5;
  A1.dotDiv(B, C, p1, p2);  // a = (b + p1) / (c + p2)
  A2 = (B + p1) / (C + p2);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorReciprocal(Tensor& A1, Tensor& A2, Tensor& B, Tensor& C) {
  real p1 = 1.5;
  real p2 = 2.5;
  real p3 = 3.5;
  A1.reciprocalSum(B, C, p1, p2, p3);  // a = 1 / (p1 * b + p2 * c + p3)
  A2 = (B * p1 + C * p2 + p3).reciprocal();
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorSoftCrossEntropy(Tensor& A1, Tensor& A2, Tensor& B, Tensor& C) {
  A1.softCrossEntropy(B, C);  // a = -c * log(b) - (1 - c) * log(1 - b)
  A2 = -C * B.log() - (C.constant(1.0f) - C) * (B.constant(1.0f) - B).log();
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorSoftCrossEntropyBp(Tensor& A1,
                                  Tensor& A2,
                                  Tensor& B,
                                  Tensor& C) {
  A1.softCrossEntropyBp(B, C);  // a += (b - c) / (b * (1 - b))
  A2 += (B - C) / (B * (B.constant(1.0f) - B));
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTernaryBaseOp(Tensor& A1, Tensor& A2, Tensor& B, Tensor& C) {
  testTensorAdd(A1, A2, B, C);
  testTensorSub(A1, A2, B, C);
  testTensorMul(A1, A2, B, C);
  testTensorDiv(A1, A2, B, C);
  testTensorReciprocal(A1, A2, B, C);
  testTensorSoftCrossEntropyBp(A1, A2, B, C);

  testTensorSoftCrossEntropy(A1, A2, B, C);
}

TEST(Ternary, BaseOp) {
  TestTernaryMatrix<CpuMatrix> testCpu(testTernaryBaseOp<CpuMatrix>);

#ifndef PADDLE_ONLY_CPU
  TestTernaryMatrix<GpuMatrix> testGpu(testTernaryBaseOp<GpuMatrix>);
#endif
}

template<typename Tensor>
void testTensorBinaryLabelCrossEntropy(Tensor& A1,
                                       Tensor& A2,
                                       Tensor& B,
                                       Tensor& C) {
  A1.binaryLabelCrossEntropy(B, C);  // a = c > 0.5 ? -log(b) : -log(1.0 - b)
  A2 = (C > (real)0.5).condition(
    -(B.log()), -((B.constant(1.0f) - B).log()));
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorBinaryLabelCrossEntropyBp(Tensor& A1,
                                         Tensor& A2,
                                         Tensor& B,
                                         Tensor& C) {
  // a += c > 0.5 ? -1.0 / b : 1.0 / (1.0 - b)
  A1.binaryLabelCrossEntropyBp(B, C);
  A2 += (C > (real)0.5).condition(
    (B.constant(-1.0f) / B), (B.constant(1.0f) - B).reciprocal());
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorLogisticRegressionLoss(Tensor& A1,
                                      Tensor& A2,
                                      Tensor& B,
                                      Tensor& C) {
  SetTensorValue(B, 50.0f);
  SetTensorValue(B, -50.0f);
  /**
   * const T THRESHOLD = 40.0;
   * T x = (b > THRESHOLD) ? THRESHOLD : (b < -THRESHOLD)
   *                                        ? -THRESHOLD
   *                                        : b;
   * a = log(1 + exp(x)) - c * x
   */
  A1.logisticRegressionLoss(B, C);
  real THRESHOLD = 40.0;
  auto tmp = (B > THRESHOLD).condition(
    THRESHOLD, (B < -THRESHOLD).condition(-THRESHOLD, B));
  A2 = (C.constant(1.0f) + tmp.exp()).log() - C * tmp;
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorLogisticRegressionLossBp(Tensor& A1,
                                        Tensor& A2,
                                        Tensor& B,
                                        Tensor& C) {
  SetTensorValue(B, 50.0f);
  SetTensorValue(B, -50.0f);
  /**
   * const T THRESHOLD = 40.0;
   * T x = (b > THRESHOLD) ? THRESHOLD : (b < -THRESHOLD)
   *                                        ? -THRESHOLD
   *                                        : b;
   * x = exp(x); a = x / (1 + x) - c
   */
  A1.logisticRegressionLossBp(B, C);
  real THRESHOLD = 40.0;
  auto tmp = (B > THRESHOLD).condition(
    THRESHOLD, (B < -THRESHOLD).condition(-THRESHOLD, B));
  auto tmp2 = tmp.exp();
  A2 = tmp2 / (C.constant(1.0) + tmp2) - C;
  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorBiggerThan(Tensor& A1, Tensor& A2, Tensor& B, Tensor& C) {
  A1.biggerThan(B, C);  // a = (b > c) ? 1.0f : 0.0f
  A2 = (B > C).condition((real)1.0f, (real)0.0f);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorMax(Tensor& A1, Tensor& A2, Tensor& B, Tensor& C) {
  A1.max2(B, C);  // a = (b > c) ? b : c
  A2 = (B > C).condition(B, C);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTernaryCompareOp(Tensor& A1, Tensor& A2, Tensor& B, Tensor& C) {
  testTensorBinaryLabelCrossEntropyBp(A1, A2, B, C);
  testTensorBinaryLabelCrossEntropy(A1, A2, B, C);
  testTensorBiggerThan(A1, A2, B, C);
  testTensorMax(A1, A2, B, C);

  testTensorLogisticRegressionLoss(A1, A2, B, C);
  testTensorLogisticRegressionLossBp(A1, A2, B, C);
}

TEST(Ternary, CompareOp) {
  TestTernaryMatrix<CpuMatrix> testCpu(testTernaryCompareOp<CpuMatrix>);

#ifndef PADDLE_ONLY_CPU
  TestTernaryMatrix<GpuMatrix> testGpu(testTernaryCompareOp<GpuMatrix>);
#endif
}

template<typename Tensor>
void testQuaternaryAdd(Tensor& A1,
                       Tensor& A2,
                       Tensor& B,
                       Tensor& C,
                       Tensor& D) {
  // A1.add3(B, C, D, 1.5f, 2.5f, 3.5f);  // a = p1 * b + p2 * c + p3 * d
  // A2 = B * 1.5f + C * 2.5f + D * 3.5f;
  // TensorCheckEqual(A1, A2);

  /*
   * T tmp = p1 * b + p2 * c + p3 * d;
   * a += tmp * tmp
   */
  real p1 = 1.5f;
  real p2 = 2.5f;
  real p3 = 3.5f;
  A1.addSquareSum(B, C, D, p1, p2, p3);
  auto tmp = B * p1 + C * p2 + D * p3;
  A2 += tmp * tmp;
  TensorCheckEqual(A1, A2);
}

TEST(Quaternary, BaseOp) {
  TestQuaternaryMatrix<CpuMatrix> testCpu(testQuaternaryAdd<CpuMatrix>);

#ifndef PADDLE_ONLY_CPU
  TestQuaternaryMatrix<GpuMatrix> testGpu(testQuaternaryAdd<GpuMatrix>);
#endif
}

template<typename Tensor>
void testTensorBiggerThan(Tensor& A1,
                          Tensor& A2,
                          Tensor& B,
                          Tensor& C,
                          Tensor& D) {
  // a = ((b > c && d > 0.5f) || (b < c && d < 0.5f)) ? 1.0f : 0.0f);
  A1.biggerThan(B, C, D);
  A2 = ((B > C && D > (real)0.5)
        || (B < C && D < (real)0.5)).condition((real)1.0, (real)0.0);
  TensorCheckEqual(A1, A2);
}

template<typename Tensor>
void testTensorRankLoss(Tensor& A1,
                        Tensor& A2,
                        Tensor& B,
                        Tensor& C,
                        Tensor& D) {
  /**
   * const T THRESHOLD = 40.0; a = b - c;
   * a = (a > THRESHOLD)
   *         ? THRESHOLD
   *         : ((a < -THRESHOLD) ? (-THRESHOLD) : a);
   * a = log(1 + exp(a)) - a * d
   */
  A1.rankLoss(B, C, D);

  real THRESHOLD = 40.0;
  auto tmp = B - C;
  auto tmp2 = (tmp > THRESHOLD).condition(
    THRESHOLD, (tmp < -THRESHOLD).condition(-THRESHOLD, tmp));
  A2 = (D.constant(1.0f) + tmp2.exp()).log() - tmp2 * D;

  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testTensorRankLossBp(Tensor& A1,
                          Tensor& A2,
                          Tensor& B,
                          Tensor& C,
                          Tensor& D) {
  /**
   * const T THRESHOLD = 40.0; a = b - c;
   * a = (a > THRESHOLD)
   *         ? THRESHOLD
   *         : ((a < -THRESHOLD) ? (-THRESHOLD) : a);
   * a = exp(a); a = (a / (1 + a) - d)
   */
  A1.rankLossBp(B, C, D);
  real THRESHOLD = 40.0;
  auto tmp = B - C;
  auto tmp2 = (tmp > THRESHOLD).condition(
    THRESHOLD, (tmp < -THRESHOLD).condition(-THRESHOLD, tmp));
  auto tmp3 = tmp2.exp();
  A2 = tmp3 / (D.constant(1.0f) + tmp3) - D;

  TensorCheckErr(A1, A2);
}

template<typename Tensor>
void testQuaternaryCompareOp(Tensor& A1,
                             Tensor& A2,
                             Tensor& B,
                             Tensor& C,
                             Tensor& D) {
  testTensorBiggerThan(A1, A2, B, C, D);
  testTensorRankLoss(A1, A2, B, C, D);
  testTensorRankLossBp(A1, A2, B, C, D);
}

TEST(Quaternary, CompareOp) {
  TestQuaternaryMatrix<CpuMatrix> testCpu(testQuaternaryCompareOp<CpuMatrix>);

#ifndef PADDLE_ONLY_CPU
  TestQuaternaryMatrix<GpuMatrix> testGpu(testQuaternaryCompareOp<GpuMatrix>);
#endif
}
