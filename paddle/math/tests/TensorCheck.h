/**
 * test_Tensor.cpp
 *
 * Author: hedaoyuan (hedaoyuan@baidu.com)
 * Created on: 2016-06-06
 *
 * Copyright (c) Baidu.com, Inc. All Rights Reserved
 */

#include <gtest/gtest.h>
#include "paddle/math/Matrix.h"
using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

template<typename Tensor>
extern void TensorCheckEqual(const Tensor& tensor1, const Tensor& tensor2);

void TensorCheckEqual(const CpuMatrix& matrix1, const CpuMatrix& matrix2) {
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

void TensorCheckEqual(const GpuMatrix& matrix1, const GpuMatrix& matrix2) {
  CpuMatrix cpu1(matrix1.getHeight(), matrix1.getWidth());
  CpuMatrix cpu2(matrix2.getHeight(), matrix2.getWidth());
  cpu1.copyFrom(matrix1);
  cpu2.copyFrom(matrix2);
  TensorCheckEqual(cpu1, cpu2);
}

void TensorCheckErr(const CpuMatrix& matrix1, const CpuMatrix& matrix2) {
  CHECK(matrix1.getHeight() == matrix2.getHeight());
  CHECK(matrix1.getWidth() == matrix2.getWidth());
#ifndef PADDLE_TYPE_DOUBLE
  real err = 1e-5;
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

void TensorCheckErr(const GpuMatrix& matrix1, const GpuMatrix& matrix2) {
  CpuMatrix cpu1(matrix1.getHeight(), matrix1.getWidth());
  CpuMatrix cpu2(matrix2.getHeight(), matrix2.getWidth());
  cpu1.copyFrom(matrix1);
  cpu2.copyFrom(matrix2);
  TensorCheckErr(cpu1, cpu2);
}

template<class T>
void TensorCheckEqual(const CpuVectorT<T>& vector1,
                      const CpuVectorT<T>& vector2) {
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

template<class T>
void TensorCheckEqual(const GpuVectorT<T>& vector1,
                      const GpuVectorT<T>& vector2) {
  CpuVectorT<T> cpu1(vector1.getSize());
  CpuVectorT<T> cpu2(vector2.getSize());
  cpu1.copyFrom(vector1);
  cpu2.copyFrom(vector2);
  TensorCheckEqual(cpu1, cpu2);
}

int VectorCheckErr(const Vector& vector1, const Vector& vector2) {
  CHECK(vector1.getSize() == vector2.getSize());

  const real* data1 = vector1.getData();
  const real* data2 = vector2.getData();
  size_t size = vector1.getSize();
  int count = 0;
  for (size_t i = 0; i < size; i++) {
    real a = data1[i];
    real b = data2[i];
    if (fabs(a - b) > FLAGS_max_diff) {
      if ((fabsf(a - b) / fabsf(a)) > (FLAGS_max_diff / 10.0f)) {
        count++;
      }
    }
  }

  return count;
}

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

// Performance Check
#ifdef PADDLE_DISABLE_TIMER

#define CHECK_VECTORPTR(vector1, vector2)   \
    EXPECT_EQ(VectorCheckErr(vector1, vector2), 0)

#define EXPRESSION_PERFORMANCE(expression)  \
    expression;

#else

#include "paddle/utils/Stat.h"

#define CHECK_VECTORPTR(vector1, vector2)

#define EXPRESSION_PERFORMANCE(expression) \
  do {\
    char expr[30];\
    strncpy(expr, #expression, 30);\
    if (expr[29] != '\0') {\
      expr[27] = '.'; expr[28] = '.'; expr[29] = '\0';\
    }\
    expression;\
    for (int i = 0; i < 20; i++) {\
      REGISTER_TIMER(expr);\
      expression;\
    }\
    LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')\
      << *globalStat.getStat(expr);\
    globalStat.reset();\
  } while (0)

#endif

