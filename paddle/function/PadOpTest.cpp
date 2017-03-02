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

TEST(Pad, real) {
  for (size_t numSamples : {5, 32}) {
    for (size_t channels : {1, 5, 32}) {
      for (size_t imgSizeH : {5, 33, 100}) {
        for (size_t imgSizeW : {5, 32, 96}) {
          VLOG(3) << " numSamples=" << numSamples << " channels=" << channels
                  << " imgSizeH=" << imgSizeH << " imgSizeW=" << imgSizeW;

          FunctionCompare compare("Pad",
                                  FuncConfig()
                                      .set("cstart", 2)
                                      .set("cend", 3)
                                      .set("hstart", 1)
                                      .set("hend", 2)
                                      .set("wstart", 3)
                                      .set("wend", 2));
          TensorShape inDims{numSamples, channels, imgSizeH, imgSizeW};
          TensorShape outDims{
              numSamples, channels + 5, imgSizeH + 3, imgSizeW + 5};
          compare.addInputs(BufferArg(VALUE_TYPE_FLOAT, inDims));
          compare.addOutputs(BufferArg(VALUE_TYPE_FLOAT, outDims, ASSIGN_TO));
          compare.run();
        }
      }
    }
  }
}

TEST(PadGrad, real) {
  for (size_t numSamples : {5, 32}) {
    for (size_t channels : {1, 5, 32}) {
      for (size_t imgSizeH : {5, 33, 100}) {
        for (size_t imgSizeW : {5, 32, 96}) {
          VLOG(3) << " numSamples=" << numSamples << " channels=" << channels
                  << " imgSizeH=" << imgSizeH << " imgSizeW=" << imgSizeW;
          FunctionCompare compare("PadGrad",
                                  FuncConfig()
                                      .set("cstart", 2)
                                      .set("cend", 3)
                                      .set("hstart", 1)
                                      .set("hend", 2)
                                      .set("wstart", 3)
                                      .set("wend", 2));
          TensorShape inDims{numSamples, channels, imgSizeH, imgSizeW};
          TensorShape outDims{
              numSamples, channels + 5, imgSizeH + 3, imgSizeW + 5};
          compare.addInputs(BufferArg(VALUE_TYPE_FLOAT, outDims));
          compare.addOutputs(BufferArg(VALUE_TYPE_FLOAT, inDims, ASSIGN_TO));
          compare.run();
        }
      }
    }
  }
}

}  // namespace paddle
