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

#include "Function.h"
#include <gtest/gtest.h>
#include "paddle/math/SparseMatrix.h"

namespace paddle {

template <DeviceType DType>
void FunctionApi(typename Tensor<real, DType>::Matrix& output,
                 const typename Tensor<real, DType>::Matrix& input);

template <>
void FunctionApi<DEVICE_TYPE_CPU>(CpuMatrix& output, const CpuMatrix& input) {
  EXPECT_EQ(output.getHeight(), 100U);
  EXPECT_EQ(output.getWidth(), 200U);
}

template <>
void FunctionApi<DEVICE_TYPE_GPU>(GpuMatrix& output, const GpuMatrix& input) {
  EXPECT_EQ(output.getHeight(), 10U);
  EXPECT_EQ(output.getWidth(), 20U);
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

/**
 * Some tests case are used to check the consistency between the BufferArg type
 * argument received by Function and the original type argument.
 *
 * Use Case:
 *  TEST() {
 *    Matrix matrix(...);
 *    CheckBufferArg lambda = [=](const BufferArg& arg) {
 *      // check matrix and arg are equivalent
 *      EXPECT_EQ(matrix, arg);
 *    }
 *
 *   BufferArgs argments{matrix...};
 *   std::vector<CheckBufferArg> checkFunc{lambda...};
 *   testBufferArgs(argments, checkFunc);
 *  }
 */
typedef std::function<void(const BufferArg&)> CheckBufferArg;

void testBufferArgs(const BufferArgs& inputs,
                    const std::vector<CheckBufferArg>& check) {
  EXPECT_EQ(inputs.size(), check.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    check[i](inputs[i]);
  }
}

void testBufferArgs(const BufferArgs& inputs, const CheckBufferArg& check) {
  EXPECT_EQ(inputs.size(), 1U);
  check(inputs[0]);
}

TEST(Arguments, Matrix) {
  MatrixPtr matrix = Matrix::create(100, 200);
  CheckBufferArg check = [=](const BufferArg& arg) {
    EXPECT_EQ(arg.shape().ndims(), 2U);
    EXPECT_EQ(arg.shape()[0], 100U);
    EXPECT_EQ(arg.shape()[1], 200U);
    EXPECT_EQ(arg.data(), matrix->getData());

    EXPECT_EQ(arg.matrix<DEVICE_TYPE_CPU>().getHeight(), matrix->getHeight());
    EXPECT_EQ(arg.matrix<DEVICE_TYPE_CPU>().getWidth(), matrix->getWidth());
    EXPECT_EQ(arg.matrix<DEVICE_TYPE_CPU>().getData(), matrix->getData());
  };

  BufferArgs argments;
  argments.addArg(*matrix);
  std::vector<CheckBufferArg> checkFunc;
  checkFunc.push_back(check);
  testBufferArgs(argments, checkFunc);
}

TEST(Arguments, Vector) {
  VectorPtr vector = Vector::create(100, false);
  CheckBufferArg check = [=](const BufferArg& arg) {
    EXPECT_EQ(arg.shape().ndims(), 1U);
    EXPECT_EQ(arg.shape()[0], 100U);
    EXPECT_EQ(arg.data(), vector->getData());

    CpuVector inVector = arg.vector<real, DEVICE_TYPE_CPU>();
    EXPECT_EQ(inVector.getSize(), vector->getSize());
    EXPECT_EQ(inVector.getData(), vector->getData());
  };

  BufferArgs argments;
  argments.addArg(*vector);
  std::vector<CheckBufferArg> checkFunc;
  checkFunc.push_back(check);
  testBufferArgs(argments, checkFunc);
}

TEST(Arguments, CpuSparseMatrix) {
  CpuSparseMatrix sparse(200, 300, 50);
  CheckBufferArg check = [=](const BufferArg& arg) {
    EXPECT_EQ(arg.shape().ndims(), 2U);
    EXPECT_EQ(arg.shape()[0], 200U);
    EXPECT_EQ(arg.shape()[1], 300U);
    EXPECT_EQ(arg.data(), sparse.getData());
    // CHECK_EQ(arg.sparse().nnz(), 50);
    // CHECK_EQ(arg.sparse().dataFormat(), SPARSE_CSR_FORMAT);
    // CHECK_EQ(arg.sparse().dataType(), SPARSE_FLOAT_VALUE);
    EXPECT_EQ(arg.sparse().getRowBuf(), sparse.getRows());
    EXPECT_EQ(arg.sparse().getColBuf(), sparse.getCols());
  };

  BufferArgs argments;
  argments.addArg(sparse);
  std::vector<CheckBufferArg> checkFunc;
  checkFunc.push_back(check);
  testBufferArgs(argments, checkFunc);
}

TEST(Arguments, BufferArg) {
  BufferArg arg(nullptr, VALUE_TYPE_FLOAT, {1, 2, 3});
  CheckBufferArg check = [=](const BufferArg& arg) {
    EXPECT_EQ(arg.shape().ndims(), 3U);
    EXPECT_EQ(arg.shape()[0], 1U);
    EXPECT_EQ(arg.shape()[1], 2U);
    EXPECT_EQ(arg.shape()[2], 3U);
  };

  BufferArgs argments;
  argments.addArg(arg);
  testBufferArgs(argments, check);
}

}  // namespace paddle
