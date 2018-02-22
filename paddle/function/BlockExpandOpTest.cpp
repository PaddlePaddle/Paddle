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

namespace paddle {

TEST(BlockExpandForward, real) {
  for (size_t batchSize : {5}) {
    for (size_t channels : {1, 5}) {
      for (size_t inputHeight : {5, 33}) {
        for (size_t inputWidth : {5, 32}) {
          for (size_t block : {1, 3, 5}) {
            for (size_t stride : {1, 2}) {
              for (size_t padding : {0, 1}) {
                // init Test object
                std::vector<size_t> strides = {stride, stride};
                std::vector<size_t> paddings = {padding, padding};
                std::vector<size_t> blocks = {block, block};
                CpuGpuFuncCompare test("BlockExpand",
                                       FuncConfig()
                                           .set("strides", strides)
                                           .set("paddings", paddings)
                                           .set("blocks", blocks));

                size_t outputHeight =
                    1 +
                    (inputHeight + 2 * padding - block + stride - 1) / stride;
                size_t outputWidth =
                    1 +
                    (inputWidth + 2 * padding - block + stride - 1) / stride;
                TensorShape inputShape =
                    TensorShape({batchSize, channels, inputHeight, inputWidth});
                TensorShape outputShape =
                    TensorShape({batchSize,
                                 outputHeight * outputWidth,
                                 channels * block * block});
                test.addInputs(BufferArg(VALUE_TYPE_FLOAT, inputShape));
                test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, outputShape));
                // run Function
                test.run();
              }
            }
          }
        }
      }
    }
  }
}

TEST(BlockExpandBackward, real) {
  for (size_t batchSize : {5}) {
    for (size_t channels : {1, 5}) {
      for (size_t inputHeight : {5, 33}) {
        for (size_t inputWidth : {5, 32}) {
          for (size_t block : {1, 3, 5}) {
            for (size_t stride : {1, 2}) {
              for (size_t padding : {0, 1}) {
                // init Test object
                std::vector<size_t> strides = {stride, stride};
                std::vector<size_t> paddings = {padding, padding};
                std::vector<size_t> blocks = {block, block};
                CpuGpuFuncCompare test("BlockExpandGrad",
                                       FuncConfig()
                                           .set("strides", strides)
                                           .set("paddings", paddings)
                                           .set("blocks", blocks));

                size_t outputHeight =
                    1 +
                    (inputHeight + 2 * padding - block + stride - 1) / stride;
                size_t outputWidth =
                    1 +
                    (inputWidth + 2 * padding - block + stride - 1) / stride;
                TensorShape inputShape =
                    TensorShape({batchSize, channels, inputHeight, inputWidth});
                TensorShape outputShape =
                    TensorShape({batchSize,
                                 outputHeight * outputWidth,
                                 channels * block * block});
                test.addInputs(BufferArg(VALUE_TYPE_FLOAT, outputShape));
                test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, inputShape),
                                ADD_TO);
                // run Function
                test.run();
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace paddle
