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
#include "FunctionTest.h"
#include "paddle/math/Matrix.h"

using namespace paddle;  // NOLINT

void testCosSimForward(size_t height_x,
                       size_t height_y,
                       size_t width,
                       real scale) {
  FunctionCompare compare("CosSimForward", FuncConfig().set("scale", scale));

  CpuMatrix cpu_arg1(height_x, width);
  CpuMatrix gpu_arg1(height_x, width);
  CpuMatrix cpu_arg2(height_y, width);
  CpuMatrix gpu_arg2(height_y, width);
  cpu_arg1.randomizeUniform();
  gpu_arg1.copyFrom(cpu_arg1);
  cpu_arg2.randomizeUniform();
  cpu_arg2.add(-0.5);
  gpu_arg2.copyFrom(cpu_arg2);
  CpuMatrix cpu_out(height_x, 1);
  GpuMatrix gpu_out(height_x, 1);

  compare.getCpuFunction()->calc(
      {Tensor(cpu_arg1.getData(), Dims{height_x, width}),
       Tensor(cpu_arg2.getData(), Dims{height_y, width})},
      {Tensor(cpu_out.getData(), Dims{height_x, 1})},
      {});
  compare.getGpuFunction()->calc(
      {Tensor(gpu_arg1.getData(), Dims{height_x, width}),
       Tensor(gpu_arg2.getData(), Dims{height_y, width})},
      {Tensor(gpu_out.getData(), Dims{height_x, 1})},
      {});

  autotest::TensorCheckErr(cpu_out, gpu_out);
}

TEST(Matrix, cosSim) {
  for (auto height_x : {10, 100, 1000}) {
    for (auto height_y : {1, height_x}) {
      for (auto width : {10, 100, 1000}) {
        for (auto scale : {1.0, 2.0}) {
          testCosSimForward(height_x, height_y, width, scale);
        }
      }
    }
  }
}
