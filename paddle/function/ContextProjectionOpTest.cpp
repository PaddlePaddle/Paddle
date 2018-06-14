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
#include "paddle/testing/TestUtil.h"

using namespace paddle;  // NOLINT

void testMatrixProjectionForward(int context_start,
                                 size_t context_length,
                                 bool is_padding,
                                 size_t batch_size,
                                 size_t input_dim) {
  size_t pad = std::max(0, -context_start) +
               std::max(0, (int)(context_start + context_length - 1));
  if (pad == 0) is_padding = false;

  CpuGpuFuncCompare test(
      "ContextProjectionForward",
      FuncConfig()
          .set("context_length", context_length)
          .set("context_start", context_start)
          .set("begin_pad", (size_t)std::max(0, -context_start)));

  // prepare input arguments
  test.addSequence(SequenceIdArg(TensorShape{batch_size}));
  test.addInputs(
      SequenceArg(VALUE_TYPE_FLOAT, TensorShape{batch_size, input_dim}));
  if (is_padding) {  // weight
    test.addInputs(SequenceArg(VALUE_TYPE_FLOAT, TensorShape{pad, input_dim}));
  }
  test.addOutputs(
      SequenceArg(VALUE_TYPE_FLOAT,
                  TensorShape{batch_size, input_dim * context_length}),
      ADD_TO);

  // run Function
  test.run();
}

void testMatrixProjectionBackward(int context_start,
                                  size_t context_length,
                                  bool is_padding,
                                  size_t batch_size,
                                  size_t input_dim) {
  size_t pad = std::max(0, -context_start) +
               std::max(0, (int)(context_start + context_length - 1));
  if (pad == 0) is_padding = false;

  CpuGpuFuncCompare test(
      "ContextProjectionBackward",
      FuncConfig()
          .set("context_length", context_length)
          .set("context_start", context_start)
          .set("begin_pad", (size_t)std::max(0, -context_start))
          .set("is_padding", is_padding)
          .set("total_pad", pad));

  // prepare input arguments
  test.addSequence(SequenceIdArg(TensorShape{batch_size}));
  test.addInputs(SequenceArg(
      VALUE_TYPE_FLOAT, TensorShape{batch_size, input_dim * context_length}));
  test.addOutputs(
      SequenceArg(VALUE_TYPE_FLOAT, TensorShape{batch_size, input_dim}),
      ADD_TO);
  if (is_padding) {  // weight
    test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{pad, input_dim}),
                    ADD_TO);
  }

  // run Function
  test.run();
}

TEST(ContextProjection, Projection) {
  for (auto context_start : {-5, -3, -1, 0, 3}) {
    for (auto context_length : {1, 2, 5, 7}) {
      for (auto trainable_padding : {false, true}) {
        for (auto batch_size : {1, 2, 5, 20, 100}) {
          for (auto input_dim : {15, 32, 63, 128, 200}) {
            VLOG(3) << " context_start=" << context_start
                    << " context_length=" << context_length
                    << " trainable_padding=" << trainable_padding
                    << " batch_size=" << batch_size
                    << " input_dim=" << input_dim;
            testMatrixProjectionForward(context_start,
                                        context_length,
                                        trainable_padding,
                                        batch_size,
                                        input_dim);
            testMatrixProjectionBackward(context_start,
                                         context_length,
                                         trainable_padding,
                                         batch_size,
                                         input_dim);
          }
        }
      }
    }
  }
}
