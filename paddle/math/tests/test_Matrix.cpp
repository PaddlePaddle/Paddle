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
 * in Matrix.cpp.
 */

#include <gtest/gtest.h>
#include "paddle/utils/Util.h"
#include "paddle/math/BaseMatrix.h"
#include "TestUtils.h"

using namespace paddle;  // NOLINT
using namespace std;     // NOLINT

/**
 * Test member functions which prototype is
 * void (Matrix::*)(Matrix&).
 */
TEST(BaseMatrix, real) {
  typedef void (Matrix::*FunctionProto)(Matrix&);
#define MATRIXCOMPARE(function) \
  BaseMatrixCompare<0>(static_cast<FunctionProto>(&Matrix::function), true);

  MATRIXCOMPARE(softmax);

#undef MATRIXCOMPARE
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  initMain(argc, argv);
  return RUN_ALL_TESTS();
}

#endif
