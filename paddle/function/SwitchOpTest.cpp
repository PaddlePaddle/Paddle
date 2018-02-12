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

TEST(Pad, real) {
  for (size_t numSamples : {1, 4, 8, 16}) {
    for (size_t channels : {1, 4, 8, 16}) {
      for (size_t imgSizeH : {1, 4, 8, 16}) {
        for (size_t imgSizeW : {1, 4, 8, 16}) {
          VLOG(3) << " numSamples=" << numSamples << " channels=" << channels
                  << " imgSizeH=" << imgSizeH << " imgSizeW=" << imgSizeW;
          for (bool test_grad : {true, false}) {
            CpuGpuFuncCompare compare(test_grad ? "NHWC2NCHW" : "NCHW2NHWC",
                                      FuncConfig());
            TensorShape inDims{numSamples, channels, imgSizeH, imgSizeW};
            TensorShape outDims{numSamples, imgSizeH, imgSizeW, channels};
            compare.addInputs(
                BufferArg(VALUE_TYPE_FLOAT, test_grad ? outDims : inDims));
            compare.addOutputs(BufferArg(
                VALUE_TYPE_FLOAT, test_grad ? inDims : outDims, ASSIGN_TO));
            compare.run();
          }
        }
      }
    }
  }
}

}  // namespace paddle
