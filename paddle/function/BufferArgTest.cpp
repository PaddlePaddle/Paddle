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

#include "BufferArg.h"
#include <gtest/gtest.h>
#include "Function.h"
#include "paddle/math/MemoryHandle.h"
#include "paddle/math/SparseMatrix.h"

namespace paddle {

TEST(BufferTest, BufferArg) {
  TensorShape shape({8, 10});
  CpuMemoryHandle memory(shape.getElements() *
                         sizeOfValuType(VALUE_TYPE_FLOAT));
  BufferArg buffer(memory.getBuf(), VALUE_TYPE_FLOAT, shape);
  EXPECT_EQ(buffer.data(), memory.getBuf());
}

TEST(BufferTest, SequenceIdArg) {
  TensorShape shape({10});
  CpuMemoryHandle memory(shape.getElements() *
                         sizeOfValuType(VALUE_TYPE_INT32));
  SequenceIdArg buffer(memory.getBuf(), shape);
  EXPECT_EQ(buffer.data(), memory.getBuf());
  EXPECT_EQ(buffer.numSeqs(), 9);
}

TEST(BufferTest, asArgument) {
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
