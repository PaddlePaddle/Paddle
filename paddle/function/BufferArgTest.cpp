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

#include "BufferArg.h"
#include <gtest/gtest.h>
#include "paddle/math/MemoryHandle.h"

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
  EXPECT_EQ(buffer.numSeqs(), 9U);
}

}  // namespace paddle
