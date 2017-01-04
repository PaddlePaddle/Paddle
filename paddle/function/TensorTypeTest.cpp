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

#include "TensorType.h"
#include <gtest/gtest.h>

namespace paddle {

TEST(TensorType, Matrix) {
  Tensor<real, DEVICE_TYPE_CPU>::Matrix matrix(100, 200);
  EXPECT_EQ(matrix.getHeight(), 100);
  EXPECT_EQ(matrix.getWidth(), 200);
  EXPECT_EQ(matrix.getElementCnt(), 100 * 200);
  EXPECT_EQ(matrix.useGpu(), false);

  Tensor<real, DEVICE_TYPE_GPU>::Matrix testGpu(100, 200);
  EXPECT_EQ(testGpu.useGpu(), true);
}

TEST(TensorType, Vector) {
  Tensor<real, DEVICE_TYPE_CPU>::Vector cpuVector(100);
  Tensor<real, DEVICE_TYPE_GPU>::Vector gpuVector(100);
  EXPECT_EQ(cpuVector.useGpu(), false);
  EXPECT_EQ(gpuVector.useGpu(), true);
  EXPECT_EQ(cpuVector.getSize(), 100);
  EXPECT_EQ(gpuVector.getSize(), 100);

  Tensor<int, DEVICE_TYPE_CPU>::Vector cpuIVector(100);
  Tensor<int, DEVICE_TYPE_GPU>::Vector gpuIVector(100);
  EXPECT_EQ(cpuIVector.useGpu(), false);
  EXPECT_EQ(gpuIVector.useGpu(), true);
  EXPECT_EQ(cpuIVector.getSize(), 100);
  EXPECT_EQ(gpuIVector.getSize(), 100);
}

}  // namespace paddle
