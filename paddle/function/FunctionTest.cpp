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
#include "paddle/math/SparseMatrix.h"

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

TEST(BufferArgs, asArgument) {
  MatrixPtr matrix = Matrix::create(100, 200);
  VectorPtr vector = Vector::create(100, false);
  CpuSparseMatrix sparse(200, 300, 50);

  // prepare arguments
  BufferArgs argments;
  argments.addArg(*matrix);
  argments.addArg(*vector);
  argments.addArg(sparse);

  // function
  auto function = [=](const BufferArgs& inputs) {
    EXPECT_EQ(inputs.size(), 3);

    // check inputs[0]
    EXPECT_EQ(inputs[0].shape().ndims(), 2);
    EXPECT_EQ(inputs[0].shape()[0], 100);
    EXPECT_EQ(inputs[0].shape()[1], 200);
    EXPECT_EQ(inputs[0].data(), matrix->getData());

    EXPECT_EQ(inputs[0].matrix<DEVICE_TYPE_CPU>().getHeight(),
              matrix->getHeight());
    EXPECT_EQ(inputs[0].matrix<DEVICE_TYPE_CPU>().getWidth(),
              matrix->getWidth());
    EXPECT_EQ(inputs[0].matrix<DEVICE_TYPE_CPU>().getData(), matrix->getData());

    // check inputs[1]
    EXPECT_EQ(inputs[1].shape().ndims(), 1);
    EXPECT_EQ(inputs[1].shape()[0], 100);
    EXPECT_EQ(inputs[1].data(), vector->getData());
    CpuVector inVector = inputs[1].vector<real, DEVICE_TYPE_CPU>();
    EXPECT_EQ(inVector.getSize(), vector->getSize());
    EXPECT_EQ(inVector.getData(), vector->getData());

    // check inputs[2]
    EXPECT_EQ(inputs[2].shape().ndims(), 2);
    EXPECT_EQ(inputs[2].shape()[0], 200);
    EXPECT_EQ(inputs[2].shape()[1], 300);
    EXPECT_EQ(inputs[2].data(), sparse.getData());
    // CHECK_EQ(inputs[2].sparse().nnz(), 50);
    // CHECK_EQ(inputs[2].sparse().dataFormat(), SPARSE_CSR_FORMAT);
    // CHECK_EQ(inputs[2].sparse().dataType(), SPARSE_FLOAT_VALUE);
    EXPECT_EQ(inputs[2].sparse().getRowBuf(), sparse.getRows());
    EXPECT_EQ(inputs[2].sparse().getColBuf(), sparse.getCols());
  };

  // call function
  function(argments);
}

}  // namespace paddle
