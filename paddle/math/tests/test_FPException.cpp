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

/**
 * This test is about floating point calculation exception.
 * Paddle catches FE_INVALID, FE DIVBYZERO and FE_OVERFLOW exceptions.
 *
 * Some exceptions occur in the middle of a set of formulas,
 * that can be circumvented by some tricks.
 * For example,
 * calculate tanh
 *   b = 2.0 / (1.0 + exp(-2 * a)) - 1.0
 *
 * If the result of (-2 * a) is too large,
 * a FE_OVERFLOW exception occurs when calculating exp.
 * But the result of tanh is no overflow problem,
 * so we can add some tricks to prevent exp calculate an excessive value.
 *
 */

#include <gtest/gtest.h>
#include "paddle/math/Matrix.h"
#include "paddle/utils/Common.h"

using namespace paddle;  // NOLINT

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
      LOG(FATAL) << "should not reach here";
    }
  }
}

template <typename Matrix>
void testTanh(real illegal) {
  MatrixPtr A = std::make_shared<Matrix>(10, 10);
  MatrixPtr B = std::make_shared<Matrix>(10, 10);
  A->randomizeUniform();
  B->randomizeUniform();

  SetTensorValue(*A, illegal);

  A->tanh(*B);
}

template <typename Matrix>
void testSigmoid(real illegal) {
  MatrixPtr A = std::make_shared<Matrix>(10, 10);
  MatrixPtr B = std::make_shared<Matrix>(10, 10);
  A->randomizeUniform();
  B->randomizeUniform();

  SetTensorValue(*A, illegal);

  A->sigmoid(*B);
}

TEST(fp, overflow) {
  for (auto illegal : {-90.0, 90.0}) {
    LOG(INFO) << " illegal=" << illegal;
    testTanh<CpuMatrix>(illegal);
    testSigmoid<CpuMatrix>(illegal);
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);

  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  return RUN_ALL_TESTS();
}
