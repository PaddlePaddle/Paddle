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

namespace paddle {

void testRowConvFw(size_t batchSize, size_t dim, size_t contextLength) {
  FunctionCompare test("RowConv", FuncConfig());

  test.addSequence(SequenceIdArg(TensorShape{batchSize}));
  test.addInputs(SequenceArg(VALUE_TYPE_FLOAT, TensorShape{batchSize, dim}));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{contextLength, dim}));

  test.addOutputs(SequenceArg(VALUE_TYPE_FLOAT, TensorShape{batchSize, dim}),
                  ADD_TO);

  test.run();
}

void testRowConvBw(size_t batchSize, size_t dim, size_t contextLength) {
  FunctionCompare test("RowConvGrad", FuncConfig());

  test.addSequence(SequenceIdArg(TensorShape{batchSize}));
  test.addInputs(SequenceArg(VALUE_TYPE_FLOAT, TensorShape{batchSize, dim}));
  test.addInputs(SequenceArg(VALUE_TYPE_FLOAT, TensorShape{batchSize, dim}));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{contextLength, dim}));

  test.addOutputs(SequenceArg(VALUE_TYPE_FLOAT, TensorShape{batchSize, dim}),
                  ADD_TO);
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{contextLength, dim}),
                  ADD_TO);

  test.run();
}

TEST(RowConv, real) {
  // for (size_t numSamples : {17, 129}) {
  //   for (size_t dim : {16, 248}) {
  //     for (size_t context: {3, 7, 65}) {
  LOG(INFO) << "===========";
  // for (size_t numSamples : {17}) {
  //  for (size_t dim : {16}) {
  //    for (size_t context: {3}) {
  size_t numSamples = 17;
  size_t dim = 16;
  size_t context = 3;
  LOG(INFO) << " numSamples=" << numSamples << " dim=" << dim
            << " context length=" << context;
  testRowConvFw(numSamples, dim, context);
  // testRowConvBw(numSamples, dim, context);
  //     }
  //   }
  // }
}

}  // namespace paddle
