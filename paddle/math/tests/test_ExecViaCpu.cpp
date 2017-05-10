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
#include <paddle/utils/PythonUtil.h>
#include <paddle/utils/Util.h>
#include <vector>
#include "paddle/math/SparseMatrix.h"

using namespace paddle;  // NOLINT

const int height = 10;
const int width = 16;

real f(Matrix& mat1,
       const Matrix& mat2,
       IVector& vec1,
       const IVector& vec2,
       real scalar) {
  CHECK(!mat1.useGpu());
  CHECK(!mat2.useGpu());
  CHECK(!vec1.useGpu());
  CHECK(!vec2.useGpu());
  mat1.copyFrom(mat2);
  vec1.copyFrom(vec2);

  return scalar;
}

class Functor {
public:
  real operator()(Matrix& mat1,
                  const Matrix& mat2,
                  IVector& vec1,
                  const IVector& vec2,
                  real scalar) {
    a_ = f(mat1, mat2, vec1, vec2, scalar);
    return a_;
  }

private:
  real a_;
};

template <typename F>
void testWrapper(F&& f) {
  MatrixPtr cpumat1 = Matrix::create(height, width, false, /*useGpu=*/false);
  MatrixPtr cpumat2 = Matrix::create(height, width, false, /*useGpu=*/false);

  IVectorPtr cpuvec1 = IVector::create(height, /*useGpu=*/false);
  IVectorPtr cpuvec2 = IVector::create(height, /*useGpu=*/false);

  const real scalar = 1.23456;

  MatrixPtr gpumat1 = Matrix::create(height, width, false, /*useGpu=*/true);
  MatrixPtr gpumat2 = Matrix::create(height, width, false, /*useGpu=*/true);
  IVectorPtr gpuvec1 = IVector::create(height, /*useGpu=*/true);
  IVectorPtr gpuvec2 = IVector::create(height, /*useGpu=*/true);

  cpumat2->randomizeUniform();
  cpuvec2->rand(width);
  gpumat2->copyFrom(*cpumat2);
  gpuvec2->copyFrom(*cpuvec2);

  real ret = execViaCpu(f, *gpumat1, *gpumat2, *gpuvec1, *gpuvec2, 1.23456);
  EXPECT_EQ(ret, scalar);
  cpumat1->copyFrom(*gpumat1);
  cpuvec1->copyFrom(*gpuvec1);

  for (int i = 0; i < height; ++i) {
    EXPECT_EQ(cpuvec1->getElement(i), cpuvec2->getElement(i));
    for (int j = 0; j < width; ++j) {
      EXPECT_EQ(cpumat1->getElement(i, j), cpumat2->getElement(i, j));
    }
  }
  gpumat1->resize(height, 1);
  execViaCpu2(&CpuMatrix::selectElements, *gpumat1, *gpumat2, *gpuvec1);

  cpumat1->resize(height, 1);
  cpumat1->selectElements(*cpumat2, *cpuvec1);
  for (int i = 0; i < height; ++i) {
    EXPECT_EQ(cpumat1->getElement(i, 0), gpumat1->getElement(i, 0));
  }
}

#ifndef PADDLE_ONLY_CPU
TEST(ExecViaCpu, test1) {
  testWrapper(f);
  testWrapper(&f);

  auto lambda = [](Matrix& mat1,
                   const Matrix& mat2,
                   IVector& vec1,
                   const IVector& vec2,
                   real scalar) -> real {
    return f(mat1, mat2, vec1, vec2, scalar);
  };
  LOG(INFO) << "lambda is_class=" << std::is_class<decltype(lambda)>::value
            << " is_function=" << std::is_function<decltype(lambda)>::value;
  testWrapper(lambda);

  Functor functor;
  testWrapper(functor);
}
#endif
