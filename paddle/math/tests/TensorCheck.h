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

// Performance Check
#ifdef PADDLE_DISABLE_TIMER

#define EXPRESSION_PERFORMANCE(expression)  \
    expression;

#else

#include "paddle/utils/Stat.h"

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

