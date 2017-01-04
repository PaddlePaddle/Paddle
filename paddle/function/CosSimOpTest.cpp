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

TEST(Matrix, cosSimForward) {
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

void testCosSimBackward(size_t height_x,
                        size_t height_y,
                        size_t width,
                        real scale) {
  FunctionCompare compare("CosSimBackward", FuncConfig().set("scale", scale));

  CpuMatrix cpu_out_grad(height_x, 1);
  CpuMatrix cpu_out_val(height_x, 1);
  CpuMatrix cpu_in1_val(height_x, width);
  CpuMatrix cpu_in2_val(height_x, width);
  CpuMatrix cpu_in1_grad(height_x, width);
  CpuMatrix cpu_in2_grad(height_x, width);

  cpu_out_grad.randomizeUniform();
  cpu_out_val.randomizeUniform();
  cpu_in1_val.randomizeUniform();
  cpu_in2_val.randomizeUniform();
  cpu_in1_grad.randomizeUniform();
  cpu_in2_grad.randomizeUniform();

  GpuMatrix gpu_out_grad(height_x, 1);
  GpuMatrix gpu_out_val(height_x, 1);
  GpuMatrix gpu_in1_val(height_x, width);
  GpuMatrix gpu_in2_val(height_x, width);
  GpuMatrix gpu_in1_grad(height_x, width);
  GpuMatrix gpu_in2_grad(height_x, width);

  gpu_out_grad.copyFrom(cpu_out_grad);
  gpu_out_val.copyFrom(cpu_out_val);
  gpu_in1_val.copyFrom(cpu_in1_val);
  gpu_in2_val.copyFrom(cpu_in2_val);
  gpu_in1_grad.copyFrom(cpu_in1_grad);
  gpu_in2_grad.copyFrom(cpu_in2_grad);

  compare.getCpuFunction()->calc(
      {Tensor(cpu_out_grad.getData(), Dims{height_x, 1}),
       Tensor(cpu_out_val.getData(), Dims{height_x, 1}),
       Tensor(cpu_in1_val.getData(), Dims{height_x, width}),
       Tensor(cpu_in2_val.getData(), Dims{height_x, width})},
      {},
      {Tensor(cpu_in1_grad.getData(), Dims{height_x, width}),
       Tensor(cpu_in2_grad.getData(), Dims{height_x, width})});

  compare.getGpuFunction()->calc(
      {Tensor(gpu_out_grad.getData(), Dims{height_x, 1}),
       Tensor(gpu_out_val.getData(), Dims{height_x, 1}),
       Tensor(gpu_in1_val.getData(), Dims{height_x, width}),
       Tensor(gpu_in2_val.getData(), Dims{height_x, width})},
      {},
      {Tensor(gpu_in1_grad.getData(), Dims{height_x, width}),
       Tensor(gpu_in2_grad.getData(), Dims{height_x, width})});

  autotest::TensorCheckErr(cpu_in1_grad, gpu_in1_grad);
  autotest::TensorCheckErr(cpu_in2_grad, gpu_in2_grad);
}

TEST(Matrix, cosSimBackward) {
  for (auto height_x : {1, 10, 100}) {
    for (auto height_y : {1, height_x}) {
      for (auto width : {1, 10, 100}) {
        for (auto scale : {1.0, 2.0}) {
          testCosSimBackward(height_x, height_y, width, scale);
        }
      }
    }
  }
}
