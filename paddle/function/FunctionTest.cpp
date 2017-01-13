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

#include "Function.h"
#include <gtest/gtest.h>

namespace paddle {

template <DeviceType DType>
void FunctionApi(typename Tensor<real, DType>::Matrix& output,
                 const typename Tensor<real, DType>::Matrix& input);

template <>
void FunctionApi<DEVICE_TYPE_CPU>(CpuMatrix& output, const CpuMatrix& input) {
  EXPECT_EQ(output.getHeight(), 100);
  EXPECT_EQ(output.getWidth(), 200);
}

template <>
void FunctionApi<DEVICE_TYPE_GPU>(GpuMatrix& output, const GpuMatrix& input) {
  EXPECT_EQ(output.getHeight(), 10);
  EXPECT_EQ(output.getWidth(), 20);
}

template <DeviceType DType>
void Function(const BufferArgs& arguments) {
  const auto input = arguments[0].matrix<DType>();
  auto output = arguments[1].matrix<DType>();
  FunctionApi<DType>(output, input);
}

TEST(Function, BufferArgs) {
  CpuMatrix cpuInput = CpuMatrix(100, 200);
  CpuMatrix cpuOutput = CpuMatrix(100, 200);
  BufferArgs cpuArgments;
  cpuArgments.addArg(cpuInput);
  cpuArgments.addArg(cpuOutput);
  Function<DEVICE_TYPE_CPU>(cpuArgments);

  GpuMatrix gpuInput = GpuMatrix(10, 20);
  GpuMatrix gpuOutput = GpuMatrix(10, 20);
  BufferArgs gpuArgments;
  gpuArgments.addArg(gpuInput);
  gpuArgments.addArg(gpuOutput);
  Function<DEVICE_TYPE_GPU>(gpuArgments);
}

}  // namespace paddle
