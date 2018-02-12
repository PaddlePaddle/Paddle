/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include "FunctionTest.h"
#include "paddle/math/Matrix.h"

using namespace paddle;  // NOLINT

void testCosSimForward(size_t height_x,
                       size_t height_y,
                       size_t width,
                       real scale) {
  CpuGpuFuncCompare test("CosSimForward", FuncConfig().set("scale", scale));
  // prepare input arguments
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{height_x, width}));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{height_y, width}));
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{height_x, 1}),
                  ASSIGN_TO);
  // run Function
  test.run();
}

void testCosSimBackward(size_t height_x,
                        size_t height_y,
                        size_t width,
                        real scale) {
  CpuGpuFuncCompare test("CosSimBackward", FuncConfig().set("scale", scale));
  // prepare input arguments
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{height_x, 1}));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{height_x, 1}));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{height_x, width}));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{height_y, width}));
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{height_x, width}),
                  ADD_TO);
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{height_y, width}),
                  ADD_TO);
  // run Function
  test.run();
}

TEST(Matrix, cosSim) {
  for (auto height_x : {10, 100, 1000}) {
    for (auto height_y : {1, height_x}) {
      for (auto width : {10, 100, 1000}) {
        for (auto scale : {1.0, 2.0}) {
          testCosSimForward(height_x, height_y, width, scale);
          testCosSimBackward(height_x, height_y, width, scale);
        }
      }
    }
  }
}
